# ==========================================================================================
# PIE v2 - Fig. 18 + full evaluation (one Colab cell, GPU runtime).  YOLOv5x, MS-COCO val2017.
#
# Fixes over v1:
#   * features are L2-normalised before K-Means / softmax  -> soft unit maps (no 0/1 maps)
#   * mask = exactly the top 15 % of union pixels of each map (ties broken in raster order),
#     so a mask can never cover the whole union region
#   * D-RISE weights are mean-centred -> real saliency instead of noise
#   * Grad-CAM and ODAM added (gradient of the matched raw YOLOv5 prediction)
#
# FIG18_ONLY = True  -> only image 367095, makes Fig. 18 (about 5 min)
# FIG18_ONLY = False -> full evaluation + statistics + Fig. 18 on a representative pair
# ==========================================================================================
!pip -q install ultralytics pycocotools

import json, math, random, time, zipfile, urllib.request, ssl, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from scipy import stats
from tqdm.auto import tqdm

# ------------------------------------------------------------------------- CONFIG
FIG18_ONLY      = True
FIG18_IMAGE_ID  = 367095               # used when FIG18_ONLY = True
CLASS_PAIRS     = [("person", "bicycle"), ("person", "laptop"), ("car", "traffic light")]
MAX_EVAL_PAIRS_PER_CLASS = 200         # None = all
MODEL_NAME, INPUT_SIZE = "yolov5x", 640
CONF, NMS_IOU, MAX_DET = 0.25, 0.50, 100
K, TAU_S, PCT, SIGMA, THETA_M = 8, 0.05, 85, 15, 0.50
FRACTIONS = np.linspace(0, 1, 21)
DRISE_MASKS_EVAL, DRISE_MASKS_FIG, DRISE_GRID, DRISE_P = 1000, 4000, 16, 0.5
RUN_DRISE_IN_EVAL = True
BATCH, SEED = 32, 42
OUT = Path("/content/pie_eval_v2"); OUT.mkdir(parents=True, exist_ok=True)
COCO_ROOT = Path("/content/coco")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE

# ------------------------------------------------------------------------- COCO
def ensure_coco():
    img_dir, ann = COCO_ROOT / "val2017", COCO_ROOT / "annotations/instances_val2017.json"
    if not (img_dir.exists() and ann.exists()):
        COCO_ROOT.mkdir(parents=True, exist_ok=True)
        for name, url in [("val2017.zip", "https://images.cocodataset.org/zips/val2017.zip"),
                          ("ann.zip", "https://images.cocodataset.org/annotations/annotations_trainval2017.zip")]:
            z = COCO_ROOT / name
            if not z.exists():
                print("Downloading", url)
                with urllib.request.urlopen(url, context=_ctx) as r, open(z, "wb") as f:
                    shutil.copyfileobj(r, f)
            zipfile.ZipFile(z).extractall(COCO_ROOT)
    return img_dir, ann

def build_pairs(coco, limit_per_class, rng, image_ids=None):
    rows = []
    for a, b in CLASS_PAIRS:
        ca, cb = coco.getCatIds(catNms=[a])[0], coco.getCatIds(catNms=[b])[0]
        ids = sorted(set(coco.getImgIds(catIds=[ca])) & set(coco.getImgIds(catIds=[cb])))
        if image_ids is not None:
            ids = [i for i in ids if i in image_ids]
        cls_rows = []
        for iid in ids:
            A = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[ca], iscrowd=False))
            B = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cb], iscrowd=False))
            if not A or not B:
                continue
            ga, gb = max(A, key=lambda x: x["area"]), max(B, key=lambda x: x["area"])
            xy = lambda bb: [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            cls_rows.append(dict(image_id=iid, file_name=coco.loadImgs([iid])[0]["file_name"],
                                 class_i=a, class_j=b, gt_i=xy(ga["bbox"]), gt_j=xy(gb["bbox"])))
        rng.shuffle(cls_rows)
        rows += cls_rows[:limit_per_class] if limit_per_class else cls_rows
    return rows

# ------------------------------------------------------------------------- geometry
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)

def iou_many(boxes, b):
    x1 = np.maximum(boxes[:, 0], b[0]); y1 = np.maximum(boxes[:, 1], b[1])
    x2 = np.minimum(boxes[:, 2], b[2]); y2 = np.minimum(boxes[:, 3], b[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    return inter / (area + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)

def union_box(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], np.float32)

def box_mask(b, W, H):
    m = np.zeros((H, W), bool); x1, y1, x2, y2 = np.round(b).astype(int)
    m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m

def norm01(x):
    x = np.asarray(x, np.float32); x = x - x.min(); return x / (x.max() + 1e-8)

# ------------------------------------------------------------------------- detector
class Detector:
    def __init__(self):
        self.m = torch.hub.load("ultralytics/yolov5", MODEL_NAME, pretrained=True, trust_repo=True).to(DEVICE).eval()
        self.m.conf, self.m.iou, self.m.max_det = CONF, NMS_IOU, MAX_DET
        self.names = self.m.names
        self.cls_index = {v: k for k, v in self.names.items()}
        self.detect_layer = self.m.model.model.model[-1]
        self.n_passes = 0

    def detect_many(self, imgs):
        out = []
        for s in range(0, len(imgs), BATCH):
            chunk = list(imgs[s:s + BATCH])
            with torch.no_grad():
                res = self.m(chunk, size=INPUT_SIZE).xyxy
            self.n_passes += len(chunk)
            for r in res:
                r = r.cpu().numpy()
                out.append([dict(box=d[:4].astype(np.float32), score=float(d[4]), cls=self.names[int(d[5])]) for d in r])
        return out

    def detect(self, img):
        return self.detect_many([img])[0]

    def _letterbox(self, img):
        H, W = img.shape[:2]; r = min(INPUT_SIZE / H, INPUT_SIZE / W)
        nw, nh = int(round(W * r)), int(round(H * r)); top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
        pad = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        pad[top:top+nh, left:left+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(pad).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
        return x, dict(r=r, top=top, left=left, nw=nw, nh=nh, W=W, H=H)

    def _forward(self, x, keep_grad):
        box = {}
        def hook(mod, inp):
            maps = [t for t in inp[0] if torch.is_tensor(t) and t.ndim == 4]
            f = min(maps, key=lambda t: abs(INPUT_SIZE / t.shape[-1] - 16))      # PANet P4, stride 16
            if keep_grad:
                f.retain_grad()
            box["f"] = f
        h = self.detect_layer.register_forward_pre_hook(hook)
        try:
            if keep_grad:
                x = x.clone().requires_grad_(True)
                with torch.enable_grad():
                    y = self.m.model(x)
            else:
                with torch.no_grad():
                    y = self.m.model(x)
        finally:
            h.remove()
        pred = y[0] if isinstance(y, (list, tuple)) else y
        return pred[0], box["f"]

    def features(self, img):
        x, g = self._letterbox(img)
        _, f = self._forward(x, keep_grad=False)
        return f[0].float().detach(), g

    def gradient_maps(self, img, target):
        """Grad-CAM and ODAM for one detection, from the raw (pre-NMS) YOLOv5 prediction
        that matches the target box and class."""
        x, g = self._letterbox(img)
        pred, f = self._forward(x, keep_grad=True)
        p = pred.detach().float().cpu().numpy()
        xyxy = np.stack([p[:, 0]-p[:, 2]/2, p[:, 1]-p[:, 3]/2, p[:, 0]+p[:, 2]/2, p[:, 1]+p[:, 3]/2], 1)
        c = self.cls_index[target["cls"]]
        tb = target["box"] * g["r"] + np.array([g["left"], g["top"], g["left"], g["top"]])
        sc = p[:, 4] * p[:, 5 + c]
        idx = int(np.argmax(iou_many(xyxy, tb) * sc))
        score = pred[idx, 4] * pred[idx, 5 + c]
        score.backward()
        grad, feat = f.grad[0].float(), f[0].detach().float()
        gradcam = torch.relu((grad.mean((1, 2))[:, None, None] * feat).sum(0)).cpu().numpy()
        odam = torch.relu((grad * feat).sum(0)).cpu().numpy()
        self.m.model.zero_grad(set_to_none=True)
        return norm01(to_pixels(gradcam, g)), norm01(to_pixels(odam, g))

def to_pixels(fmap, g):
    up = cv2.resize(fmap.astype(np.float32), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    crop = up[g["top"]:g["top"]+g["nh"], g["left"]:g["left"]+g["nw"]]
    return cv2.resize(crop, (g["W"], g["H"]), interpolation=cv2.INTER_LINEAR)

# ------------------------------------------------------------------------- matching (Alg. 4)
def match_pair(dets, di, dj):
    ci = [None] + [k for k, d in enumerate(dets) if d["cls"] == di["cls"] and iou(d["box"], di["box"]) >= THETA_M]
    cj = [None] + [k for k, d in enumerate(dets) if d["cls"] == dj["cls"] and iou(d["box"], dj["box"]) >= THETA_M]
    best, key = (None, None), (-1.0, -1.0)
    for a in ci:
        for b in cj:
            if a is not None and a == b:
                continue
            s = (iou(dets[a]["box"], di["box"]) if a is not None else 0) + (iou(dets[b]["box"], dj["box"]) if b is not None else 0)
            c = (dets[a]["score"] if a is not None else 0) + (dets[b]["score"] if b is not None else 0)
            if (s, c) > key:
                key, best = (s, c), (a, b)
    a, b = best
    return (dets[a]["score"] if a is not None else 0.0), (dets[b]["score"] if b is not None else 0.0)

def pick_detection(dets, cls, gt):
    cand = [d for d in dets if d["cls"] == cls and iou(d["box"], gt) >= 0.5]
    return max(cand, key=lambda d: iou(d["box"], gt)) if cand else None

# ------------------------------------------------------------------------- PIE
def top_fraction_mask(m, omega, frac):
    """Exactly the top `frac` of union pixels by map value; ties broken in raster order."""
    ys, xs = np.nonzero(omega); v = m[ys, xs]
    n = max(1, int(math.ceil(frac * len(v))))
    keep = np.argsort(-v, kind="stable")[:n]
    out = np.zeros_like(omega); out[ys[keep], xs[keep]] = True
    return out

def pie_units(det, img, di, dj):
    F_, g = det.features(img); C, h, w = F_.shape
    Fn = F_ / F_.norm(dim=0, keepdim=True).clamp_min(1e-6)                    # L2-normalised features
    U = union_box(di["box"], dj["box"]); H, W = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx = ((xs + 0.5) * INPUT_SIZE / w - g["left"]) / g["r"]; cy = ((ys + 0.5) * INPUT_SIZE / h - g["top"]) / g["r"]
    inside = (cx >= U[0]) & (cx <= U[2]) & (cy >= U[1]) & (cy <= U[3])
    V = Fn.permute(1, 2, 0)[torch.from_numpy(inside).to(Fn.device)]
    if V.shape[0] < K:
        return None
    mu = torch.tensor(KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(V.cpu().numpy()).cluster_centers_,
                      device=Fn.device, dtype=torch.float32)
    mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-6)
    d2 = torch.cdist(Fn.permute(1, 2, 0).reshape(-1, C), mu) ** 2              # Eq. 21
    Uk = torch.softmax(-d2 / TAU_S, dim=1).T.reshape(K, h, w).cpu().numpy()    # Eq. 22
    omega = box_mask(U, W, H)
    maps = np.stack([np.where(omega, to_pixels(Uk[q], g), 0) for q in range(K)])
    blur = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(SIGMA)))
    masks = np.stack([top_fraction_mask(maps[q], omega, 1 - PCT / 100) for q in range(K)])   # Eqs. 25-26
    post = det.detect_many([np.where(masks[q][..., None], blur, img).astype(np.uint8) for q in range(K)])
    pij = np.array([match_pair(p, di, dj) for p in post])
    s_i, s_j = di["score"], dj["score"]
    ce = s_i * s_j - pij[:, 0] * pij[:, 1]                                      # Eq. 34
    fo = -(s_j * (pij[:, 0] - s_i) + s_i * (pij[:, 1] - s_j))
    act = np.array([maps[q][omega].mean() for q in range(K)])
    return dict(maps=maps, masks=masks, omega=omega, U=U, blur=blur, ce=ce, fo=fo, act=act, post=pij)

# ------------------------------------------------------------------------- deletion / insertion
def unit_pixel_order(scores, maps, masks, omega):
    order = np.argsort(-scores, kind="stable"); rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    ys, xs = np.nonzero(omega); m = masks[:, ys, xs]; covered = m.any(0)
    u = np.where(covered, np.where(m, rank[:, None], 999).argmin(0), maps[:, ys, xs].argmax(0))
    keys = np.lexsort((ys * omega.shape[1] + xs, -maps[u, ys, xs], rank[u], ~covered))
    return ys[keys], xs[keys]

def map_pixel_order(rel, omega):
    ys, xs = np.nonzero(omega); keys = np.lexsort((ys * omega.shape[1] + xs, -rel[ys, xs]))
    return ys[keys], xs[keys]

_trapz = getattr(np, "trapezoid", None) or np.trapz

def morf_and_insertion(det, img, blur, omega, order, di, dj):
    ys, xs = order; n = len(ys); S0 = di["score"] * dj["score"]
    base = np.where(omega[..., None], blur, img).astype(np.uint8); dele, ins = [], []
    for a in FRACTIONS:
        k = int(np.floor(a * n)); m = np.zeros(omega.shape, bool); m[ys[:k], xs[:k]] = True
        dele.append(np.where(m[..., None], blur, img).astype(np.uint8))
        ins.append(np.where(m[..., None], img, base).astype(np.uint8))
    res = det.detect_many(dele + ins); L = len(FRACTIONS)
    sd = np.array([np.prod(match_pair(r, di, dj)) for r in res[:L]])
    si = np.array([np.prod(match_pair(r, di, dj)) for r in res[L:]])
    return float(_trapz(sd / max(S0, 1e-8), FRACTIONS)), float(_trapz(si / max(S0, 1e-8), FRACTIONS))

# ------------------------------------------------------------------------- D-RISE (mean-centred)
def drise_pair(det, img, di, dj, n_masks, rng):
    H, W = img.shape[:2]; cell = np.ceil(np.array([H, W]) / DRISE_GRID).astype(int); up = (DRISE_GRID + 1) * cell
    sal = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
    ws = [[], []]; ms_all = []
    for s in range(0, n_masks, BATCH):
        ms = []
        for _ in range(min(BATCH, n_masks - s)):
            g = (rng.random((DRISE_GRID, DRISE_GRID)) < DRISE_P).astype(np.float32)
            m = cv2.resize(g, (up[1], up[0]), interpolation=cv2.INTER_LINEAR)
            oy, ox = rng.integers(0, cell[0]), rng.integers(0, cell[1]); ms.append(m[oy:oy+H, ox:ox+W])
        res = det.detect_many([(img * m[..., None]).astype(np.uint8) for m in ms])
        for m, r in zip(ms, res):
            for t, tgt in enumerate((di, dj)):
                ws[t].append(max([iou(d["box"], tgt["box"]) * d["score"] for d in r if d["cls"] == tgt["cls"]] or [0.0]))
            ms_all.append(m.astype(np.float16))
    for t in range(2):
        w = np.asarray(ws[t]); w = w - w.mean()                                     # mean-centred weights
        for wi, m in zip(w, ms_all):
            sal[t] += wi * m.astype(np.float32)
        sal[t] = norm01(np.maximum(sal[t], 0))
    return sal[0], sal[1]

# ------------------------------------------------------------------------- one pair
def evaluate_pair(det, img, row, rng, with_drise):
    dets = det.detect(img)
    di, dj = pick_detection(dets, row["class_i"], row["gt_i"]), pick_detection(dets, row["class_j"], row["gt_j"])
    if di is None or dj is None:
        return None, "target not detected"
    P = pie_units(det, img, di, dj)
    if P is None:
        return None, "union too small"
    out = dict(image_id=row["image_id"], class_pair=f'{row["class_i"]}+{row["class_j"]}',
               s_i=di["score"], s_j=dj["score"], ce_max=float(P["ce"].max()),
               tau_ce_fo=float(stats.kendalltau(P["ce"], P["fo"])[0]))
    for name, sc in [("PIE", P["ce"]), ("First-order", P["fo"]), ("Activation-ranked", P["act"])]:
        out[f"morf_{name}"], out[f"ins_{name}"] = morf_and_insertion(
            det, img, P["blur"], P["omega"], unit_pixel_order(sc, P["maps"], P["masks"], P["omega"]), di, dj)
    gi, oi = det.gradient_maps(img, di); gj, oj = det.gradient_maps(img, dj)
    maps = {"Grad-CAM": 0.5 * (gi + gj), "ODAM": 0.5 * (oi + oj)}               # merged as in Eq. 2
    extra = dict(gradcam=(gi, gj), odam=(oi, oj))
    if with_drise:
        ri, rj = drise_pair(det, img, di, dj, DRISE_MASKS_EVAL, rng)
        maps["D-RISE"] = 0.5 * (ri + rj); extra["drise"] = (ri, rj)
    for name, rel in maps.items():
        out[f"morf_{name}"], out[f"ins_{name}"] = morf_and_insertion(
            det, img, P["blur"], P["omega"], map_pixel_order(rel, P["omega"]), di, dj)
    return out, (P, di, dj, extra)

# ------------------------------------------------------------------------- statistics
def boot_ci(x, n=5000, rng=np.random.default_rng(SEED)):
    x = np.asarray(x); return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

def report(df):
    methods = [c[5:] for c in df.columns if c.startswith("morf_")]
    tab = []
    for m in methods:
        for metric in ("morf", "ins"):
            x = df[f"{metric}_{m}"].dropna(); lo, hi = boot_ci(x)
            tab.append(dict(method=m, metric="MoRF-AUC (lower better)" if metric == "morf" else "Insertion AUC (higher better)",
                            mean=x.mean(), ci_low=lo, ci_high=hi, n=len(x)))
    tab = pd.DataFrame(tab); tests = []
    for m in methods:
        if m == "PIE":
            continue
        for metric in ("morf", "ins"):
            a, b = df[f"{metric}_PIE"], df[f"{metric}_{m}"]; ok = a.notna() & b.notna()
            d = (b - a)[ok] if metric == "morf" else (a - b)[ok]
            p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(a[ok], b[ok]).pvalue
            tests.append(dict(comparison=f"PIE vs {m}", metric=metric, mean_diff_PIE_better=d.mean(),
                              ci=tuple(np.round(boot_ci(d), 4)), d_z=d.mean() / (d.std(ddof=1) + 1e-12),
                              PIE_better_on=f"{100*(d > 0).mean():.1f}%", p=p))
    t = pd.DataFrame(tests); t["p_holm"] = holm(t["p"])
    txt = "\n".join([f"Evaluated pairs: {len(df)}", "", tab.round(4).to_string(index=False), "",
                     t.round(4).to_string(index=False), "",
                     f"Mean Kendall tau (CE vs first-order): {df['tau_ce_fo'].mean():.3f}", "",
                     "Per class pair (MoRF-AUC):",
                     df.groupby("class_pair")[[f"morf_{m}" for m in methods]].mean().round(4).to_string()])
    (OUT / "stats_report.txt").write_text(txt); print(txt)
    tab.to_csv(OUT / "summary_table.csv", index=False); t.to_csv(OUT / "significance_tests.csv", index=False)

# ------------------------------------------------------------------------- Fig. 18
def make_fig18(det, img, row, P, di, dj, extra, rng, rule_text):
    if "drise" in extra and DRISE_MASKS_FIG <= DRISE_MASKS_EVAL:
        ri, rj = extra["drise"]
    else:
        ri, rj = drise_pair(det, img, di, dj, DRISE_MASKS_FIG, rng)
    (gi, gj), (oi, oj) = extra["gradcam"], extra["odam"]
    order = np.argsort(-P["ce"]); top = order[0]; U = P["U"]; ci, cj = di["cls"], dj["cls"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig = plt.figure(figsize=(12, 8.2)); gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.08)

    def show(pos, title, heat=None, mask=None, boxes=(), union=False):
        ax = fig.add_subplot(gs[pos]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        if heat is not None:
            ax.imshow(heat, cmap="jet", alpha=0.5)
        if mask is not None:
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.65)
        for b, col, txt in boxes:
            ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=col, lw=2))
            ax.text(b[0], b[1]-3, txt, fontsize=7.5, va="bottom", bbox=dict(fc=col, ec="none", pad=1))
        if union:
            ax.add_patch(Rectangle((U[0], U[1]), U[2]-U[0], U[3]-U[1], fill=False, ec="white", lw=1.5, ls="--"))
        ax.set_title(title, weight="bold", fontsize=9)
        return ax

    show((0, 0), "(a) Target pair", boxes=[(di["box"], "#17BECF", f'{ci} {di["score"]:.2f}'),
                                           (dj["box"], "#F1C40F", f'{cj} {dj["score"]:.2f}')])
    ax = show((0, 1), r"(b) Union region $\Omega_{ij}$")
    ax.add_patch(Rectangle((U[0], U[1]), U[2]-U[0], U[3]-U[1], fc="#C000C0", alpha=0.28, ec="#C000C0", lw=2))
    b = fig.add_subplot(gs[0, 2:]); vals = P["ce"][order][::-1]
    b.barh(range(K), vals, color=["#C0392B" if i == K - 1 else "#2E6DA4" for i in range(K)], height=0.65)
    b.set_yticks(range(K)); b.set_yticklabels([f"U{q+1}" for q in order][::-1]); b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("Confidence effect $CE_k$"); b.set_title("(c) PIE: units ranked by $CE_k$", weight="bold"); b.grid(axis="x", alpha=0.3)
    for s in ("top", "right"): b.spines[s].set_visible(False)
    show((1, 0), f"(d) Grad-CAM: {ci}", heat=gi); show((1, 1), f"(e) Grad-CAM: {cj}", heat=gj)
    show((1, 2), f"(f) ODAM: {ci}", heat=oi);     show((1, 3), f"(g) ODAM: {cj}", heat=oj)
    show((2, 0), f"(h) D-RISE: {ci}", heat=ri);   show((2, 1), f"(i) D-RISE: {cj}", heat=rj)
    part = np.where(P["omega"], P["maps"].argmax(0), -1).astype(float); part[part < 0] = np.nan
    ax = show((2, 2), "(j) PIE: latent units in $\\Omega_{ij}$", union=True); ax.imshow(part, cmap="tab10", alpha=0.55, vmin=0, vmax=9)
    show((2, 3), f"(k) PIE: top-ranked unit U{top+1}", mask=P["masks"][top], union=True)
    fig.savefig(OUT / "fig18_person_laptop.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig18_person_laptop.png", dpi=300, bbox_inches="tight"); plt.show()
    S = di["score"] * dj["score"]
    cap = (r"\caption{Person--laptop pair (MS-COCO image " + str(row["image_id"]) + r")" + rule_text + r". (a) Detected pair and "
           r"(b) union region $\Omega_{ij}$. (c) The $K=8$ latent units of \ours{} ranked by $CE_k$. (d)--(i) Grad-CAM, ODAM, and "
           r"D-RISE produce one map per object. (j) Latent units inside $\Omega_{ij}$ and (k) the mask of the top-ranked unit, "
           r"which lowers the joint confidence by " + f"{P['ce'].max():.3f}" + r" of $S_{ij}=" + f"{S:.3f}" + r"$.}")
    (OUT / "fig18_caption.tex").write_text(cap); print("\n" + cap)
    pd.DataFrame(dict(unit=[f"U{q+1}" for q in range(K)], CE_k=P["ce"], post_s_i=P["post"][:, 0], post_s_j=P["post"][:, 1],
                      first_order=P["fo"])).sort_values("CE_k", ascending=False).to_csv(OUT / "fig18_unit_effects.csv", index=False)

# ------------------------------------------------------------------------- MAIN
t0 = time.time()
img_dir, ann = ensure_coco(); coco = COCO(str(ann)); det = Detector(); rng = np.random.default_rng(SEED)

if FIG18_ONLY:
    row = next(p for p in build_pairs(coco, None, rng, image_ids={FIG18_IMAGE_ID}) if p["class_j"] == "laptop")
    img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
    r, info = evaluate_pair(det, img, row, rng, with_drise=False)
    assert r is not None, info
    print(pd.Series(r).to_string())
    P, di, dj, extra = info
    make_fig18(det, img, row, P, di, dj, extra, rng, "")
else:
    pairs = build_pairs(coco, MAX_EVAL_PAIRS_PER_CLASS, rng)
    results, skipped, rows_pl = [], {}, {}
    for row in tqdm(pairs, desc="evaluation"):
        img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
        try:
            r, info = evaluate_pair(det, img, row, rng, with_drise=RUN_DRISE_IN_EVAL)
        except Exception as e:
            r, info = None, f"error: {type(e).__name__}"
        if r is None:
            skipped[info] = skipped.get(info, 0) + 1; continue
        results.append(r)
        if r["class_pair"] == "person+laptop":
            rows_pl[row["image_id"]] = row
    df = pd.DataFrame(results); df.to_csv(OUT / "per_pair_results.csv", index=False)
    print(f"\nEvaluated {len(df)} / {len(pairs)} pairs. Skipped (reported): {skipped}")
    report(df)
    pl = df[df.class_pair == "person+laptop"]
    fid = int(pl.iloc[(pl["morf_PIE"] - pl["morf_PIE"].median()).abs().argsort().iloc[0]]["image_id"])
    row = rows_pl[fid]; img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
    _, (P, di, dj, extra) = evaluate_pair(det, img, row, rng, with_drise=False)
    make_fig18(det, img, row, P, di, dj, extra, rng,
               r", the pair whose \ours{} MoRF-AUC is closest to the median of the evaluated person--laptop pairs")

json.dump(dict(model=MODEL_NAME, input=INPUT_SIZE, conf=CONF, nms=NMS_IOU, K=K, tau_s=TAU_S, percentile=PCT,
               sigma=SIGMA, theta_m=THETA_M, l2_normalised_features=True, drise_masks_eval=DRISE_MASKS_EVAL,
               drise_masks_fig=DRISE_MASKS_FIG, seed=SEED, minutes=(time.time()-t0)/60),
          open(OUT / "run_config.json", "w"), indent=2)
shutil.make_archive("/content/pie_eval_v2", "zip", OUT)
from google.colab import files; files.download("/content/pie_eval_v2.zip")