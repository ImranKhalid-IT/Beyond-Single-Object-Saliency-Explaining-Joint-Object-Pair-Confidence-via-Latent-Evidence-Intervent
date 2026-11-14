# ==========================================================================================
# PIE - full evaluation + analysis (one Colab cell, GPU).  YOLOv5x, MS-COCO val2017.
#
# What it does, with the settings written in the paper (Table 5, Eqs. 14-42):
#   1. builds the co-occurring pair list (person+bicycle, person+laptop, car+traffic light)
#   2. (optional) selects K, sigma, mask percentile on COCO *train2017* pairs only (Table 6)
#   3. runs PIE on every evaluation pair and ranks the units three ways:
#        CE_k (PIE, Eq. 34) | first-order FO_k (no cross term) | activation
#   4. evaluates every ranking + D-RISE with the same MoRF deletion (Eqs. 38-42) and insertion
#   5. writes per-pair CSVs and a statistics report (mean, 95% bootstrap CI, paired Wilcoxon,
#      Holm correction, effect size), and makes Fig. 18 on a REPRESENTATIVE person-laptop pair
#      chosen by a fixed rule (median MoRF of PIE) - not the best-looking one.
#
# Honest-evaluation rules built in:
#   * hyperparameters are never tuned on evaluation pairs
#   * every method uses the same pixel budget, blur, matching and deletion fractions
#   * nothing is filtered out after seeing results; skipped pairs are counted and reported
# ==========================================================================================
!pip -q install ultralytics pycocotools

import json, math, random, time, zipfile, urllib.request, ssl
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
CLASS_PAIRS   = [("person", "bicycle"), ("person", "laptop"), ("car", "traffic light")]
MAX_EVAL_PAIRS_PER_CLASS = 200        # None = all pairs. Start with 200 to check timing.
RUN_SELECTION = False                 # True = redo Table 6 selection on train2017 pairs
SELECTION_PAIRS = 500
MODEL_NAME, INPUT_SIZE = "yolov5x", 640
CONF, NMS_IOU, MAX_DET = 0.25, 0.50, 100
K, TAU_S, PCT, SIGMA, THETA_M = 8, 0.05, 85, 15, 0.50
FRACTIONS = np.linspace(0, 1, 21)     # deletion / insertion budgets, shared by all methods
DRISE_MASKS, DRISE_GRID, DRISE_P = 1000, 16, 0.5
RUN_DRISE = True                      # D-RISE is the slow part (2 x DRISE_MASKS passes / pair)
BATCH = 32
SEED = 42
FIG18_IMAGE_ID = None                 # None = representative pair chosen by the fixed rule
OUT = Path("/content/pie_eval"); OUT.mkdir(parents=True, exist_ok=True)
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
                    f.write(r.read())
            zipfile.ZipFile(z).extractall(COCO_ROOT)
    return img_dir, ann

def train_image(file_name):
    """Fetch single train2017 images on demand (avoids the 18 GB zip)."""
    p = COCO_ROOT / "train2017_subset" / file_name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://images.cocodataset.org/train2017/{file_name}", context=_ctx) as r:
            p.write_bytes(r.read())
    return p

def build_pairs(coco, limit_per_class, rng):
    """All (image, object_i, object_j) ground-truth pairs of the three class pairs.
    One pair per image and class pair: the two largest instances (fixed rule, decided in advance)."""
    rows = []
    for a, b in CLASS_PAIRS:
        ca, cb = coco.getCatIds(catNms=[a])[0], coco.getCatIds(catNms=[b])[0]
        ids = sorted(set(coco.getImgIds(catIds=[ca])) & set(coco.getImgIds(catIds=[cb])))
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

def union_box(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], np.float32)

def box_mask(b, W, H):
    m = np.zeros((H, W), bool); x1, y1, x2, y2 = np.round(b).astype(int)
    m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return m

# ------------------------------------------------------------------------- detector
class Detector:
    def __init__(self):
        self.m = torch.hub.load("ultralytics/yolov5", MODEL_NAME, pretrained=True, trust_repo=True).to(DEVICE).eval()
        self.m.conf, self.m.iou, self.m.max_det = CONF, NMS_IOU, MAX_DET
        self.names = self.m.names
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

    def features(self, img):
        H, W = img.shape[:2]; r = min(INPUT_SIZE / H, INPUT_SIZE / W)
        nw, nh = int(round(W * r)), int(round(H * r)); top, left = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
        pad = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        pad[top:top+nh, left:left+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(pad).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
        box = {}
        def hook(mod, inp):
            maps = [t for t in inp[0] if torch.is_tensor(t) and t.ndim == 4]
            box["f"] = min(maps, key=lambda t: abs(INPUT_SIZE / t.shape[-1] - 16)).detach()   # PANet P4
        h = self.detect_layer.register_forward_pre_hook(hook)
        with torch.no_grad():
            self.m.model(x)
        h.remove()
        return box["f"][0].float(), dict(r=r, top=top, left=left, nw=nw, nh=nh, W=W, H=H)

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
def pie_units(det, img, di, dj, k=K, pct=PCT, sigma=SIGMA):
    F_, g = det.features(img); C, h, w = F_.shape
    U = union_box(di["box"], dj["box"]); H, W = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    cx = ((xs + 0.5) * INPUT_SIZE / w - g["left"]) / g["r"]; cy = ((ys + 0.5) * INPUT_SIZE / h - g["top"]) / g["r"]
    inside = (cx >= U[0]) & (cx <= U[2]) & (cy >= U[1]) & (cy <= U[3])
    V = F_.permute(1, 2, 0)[torch.from_numpy(inside).to(F_.device)]
    if V.shape[0] < k:
        return None
    mu = torch.tensor(KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(V.cpu().numpy()).cluster_centers_,
                      device=F_.device, dtype=torch.float32)
    d2 = torch.cdist(F_.permute(1, 2, 0).reshape(-1, C), mu) ** 2
    Uk = torch.softmax(-d2 / TAU_S, dim=1).T.reshape(k, h, w).cpu().numpy()
    omega = box_mask(U, W, H)
    maps = np.stack([np.where(omega, to_pixels(Uk[q], g), 0) for q in range(k)])
    blur = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(sigma)))
    masks = np.stack([omega & (maps[q] >= np.percentile(maps[q][omega], pct)) for q in range(k)])
    post = det.detect_many([np.where(masks[q][..., None], blur, img).astype(np.uint8) for q in range(k)])
    pij = np.array([match_pair(p, di, dj) for p in post])
    s_i, s_j = di["score"], dj["score"]
    ce = s_i * s_j - pij[:, 0] * pij[:, 1]                                  # Eq. 34
    fo = -(s_j * (pij[:, 0] - s_i) + s_i * (pij[:, 1] - s_j))              # Eq. 35 without cross term
    act = np.array([maps[q][omega].mean() for q in range(k)])              # activation strength
    return dict(maps=maps, masks=masks, omega=omega, U=U, blur=blur, ce=ce, fo=fo, act=act, post=pij)

def unit_pixel_order(rank_scores, maps, masks, omega):
    """Eq. 38: masked pixels first, by unit rank, then by map value; raster tie-break."""
    order = np.argsort(-rank_scores, kind="stable"); rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    ys, xs = np.nonzero(omega)
    m = masks[:, ys, xs]; covered = m.any(0)
    u = np.where(covered, np.where(m, rank[:, None], 999).argmin(0), maps[:, ys, xs].argmax(0))
    keys = np.lexsort((ys * omega.shape[1] + xs, -maps[u, ys, xs], rank[u], ~covered))
    return ys[keys], xs[keys]

def map_pixel_order(rel, omega):
    ys, xs = np.nonzero(omega)
    keys = np.lexsort((ys * omega.shape[1] + xs, -rel[ys, xs]))
    return ys[keys], xs[keys]

_trapz = getattr(np, "trapezoid", None) or np.trapz

def morf_and_insertion(det, img, blur, omega, order, di, dj):
    ys, xs = order; n = len(ys); S0 = di["score"] * dj["score"]
    dele, ins = [], []
    for a in FRACTIONS:
        k = int(np.floor(a * n)); m = np.zeros(omega.shape, bool); m[ys[:k], xs[:k]] = True
        dele.append(np.where(m[..., None], blur, img).astype(np.uint8))                       # Eq. 40
        base = np.where(omega[..., None], blur, img).astype(np.uint8)
        ins.append(np.where(m[..., None], img, base).astype(np.uint8))
    res = det.detect_many(dele + ins)
    sd = np.array([np.prod(match_pair(r, di, dj)) for r in res[:len(FRACTIONS)]])
    si = np.array([np.prod(match_pair(r, di, dj)) for r in res[len(FRACTIONS):]])
    return float(_trapz(sd / max(S0, 1e-8), FRACTIONS)), float(_trapz(si / max(S0, 1e-8), FRACTIONS))

# ------------------------------------------------------------------------- D-RISE
def drise_pair(det, img, di, dj, rng):
    H, W = img.shape[:2]; cell = np.ceil(np.array([H, W]) / DRISE_GRID).astype(int); up = (DRISE_GRID + 1) * cell
    ms = []
    for _ in range(DRISE_MASKS):
        g = (rng.random((DRISE_GRID, DRISE_GRID)) < DRISE_P).astype(np.float32)
        m = cv2.resize(g, (up[1], up[0]), interpolation=cv2.INTER_LINEAR)
        oy, ox = rng.integers(0, cell[0]), rng.integers(0, cell[1]); ms.append(m[oy:oy+H, ox:ox+W])
    res = det.detect_many([(img * m[..., None]).astype(np.uint8) for m in ms])
    sal = []
    for t in (di, dj):
        s = np.zeros((H, W), np.float32)
        for m, r in zip(ms, res):
            s += max([iou(d["box"], t["box"]) * d["score"] for d in r if d["cls"] == t["cls"]] or [0.0]) * m
        s -= s.min(); sal.append(s / (s.max() + 1e-8))
    return sal[0], sal[1], 0.5 * (sal[0] + sal[1])                         # merged as in Eq. 2

# ------------------------------------------------------------------------- one pair
def evaluate_pair(det, img, row, rng, with_drise=RUN_DRISE, k=K, pct=PCT, sigma=SIGMA):
    dets = det.detect(img)
    di, dj = pick_detection(dets, row["class_i"], row["gt_i"]), pick_detection(dets, row["class_j"], row["gt_j"])
    if di is None or dj is None:
        return None, "target not detected"
    P = pie_units(det, img, di, dj, k, pct, sigma)
    if P is None:
        return None, "union too small"
    out = dict(image_id=row["image_id"], class_pair=f'{row["class_i"]}+{row["class_j"]}',
               s_i=di["score"], s_j=dj["score"], ce_max=float(P["ce"].max()),
               tau_ce_fo=float(stats.kendalltau(P["ce"], P["fo"])[0]))
    for name, sc in [("PIE", P["ce"]), ("First-order", P["fo"]), ("Activation-ranked", P["act"])]:
        out[f"morf_{name}"], out[f"ins_{name}"] = morf_and_insertion(
            det, img, P["blur"], P["omega"], unit_pixel_order(sc, P["maps"], P["masks"], P["omega"]), di, dj)
    if with_drise:
        _, _, merged = drise_pair(det, img, di, dj, rng)
        out["morf_D-RISE"], out["ins_D-RISE"] = morf_and_insertion(
            det, img, P["blur"], P["omega"], map_pixel_order(merged, P["omega"]), di, dj)
    return out, (P, di, dj)

# ------------------------------------------------------------------------- statistics
def boot_ci(x, n=5000, rng=np.random.default_rng(SEED)):
    x = np.asarray(x); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, [2.5, 97.5])

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj

def report(df):
    methods = [c[5:] for c in df.columns if c.startswith("morf_")]
    lines = [f"Evaluated pairs: {len(df)}", ""]
    tab = []
    for m in methods:
        for metric in ("morf", "ins"):
            x = df[f"{metric}_{m}"].dropna()
            lo, hi = boot_ci(x)
            tab.append(dict(method=m, metric="MoRF-AUC (lower better)" if metric == "morf" else "Insertion AUC (higher better)",
                            mean=x.mean(), ci_low=lo, ci_high=hi, n=len(x)))
    tab = pd.DataFrame(tab); lines.append(tab.round(4).to_string(index=False)); lines.append("")
    tests = []
    for m in methods:
        if m == "PIE":
            continue
        for metric in ("morf", "ins"):
            a, b = df[f"{metric}_PIE"], df[f"{metric}_{m}"]; ok = a.notna() & b.notna()
            d = (b - a)[ok] if metric == "morf" else (a - b)[ok]      # positive = PIE better
            p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(a[ok], b[ok]).pvalue
            tests.append(dict(comparison=f"PIE vs {m}", metric=metric, mean_diff_PIE_better=d.mean(),
                              ci=tuple(np.round(boot_ci(d), 4)), d_z=d.mean() / (d.std(ddof=1) + 1e-12),
                              PIE_better_on=f"{100*(d > 0).mean():.1f}%", p=p))
    t = pd.DataFrame(tests); t["p_holm"] = holm(t["p"]); lines.append(t.round(4).to_string(index=False))
    lines += ["", f"Mean Kendall tau (CE vs first-order ranking): {df['tau_ce_fo'].mean():.3f}",
              "", "Per class pair (MoRF-AUC):",
              df.groupby("class_pair")[[f"morf_{m}" for m in methods]].mean().round(4).to_string()]
    txt = "\n".join(lines); (OUT / "stats_report.txt").write_text(txt); print(txt)
    tab.to_csv(OUT / "summary_table.csv", index=False); t.to_csv(OUT / "significance_tests.csv", index=False)
    return tab, t

# ------------------------------------------------------------------------- Fig. 18
def make_fig18(det, img, row, P, di, dj, rng):
    sal_i, sal_j, _ = drise_pair(det, img, di, dj, rng)
    order = np.argsort(-P["ce"]); top = order[0]; U = P["U"]
    def rect(ax, b, col, txt=None, ls="-"):
        ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=col, lw=2, ls=ls))
        if txt: ax.text(b[0], b[1]-3, txt, fontsize=7.5, va="bottom", bbox=dict(fc=col, ec="none", pad=1))
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for a in ax.flat[:5]: a.set_xticks([]); a.set_yticks([])
    ax[0, 0].imshow(img); ax[0, 0].set_title("(a) Target pair", weight="bold")
    rect(ax[0, 0], di["box"], "#17BECF", f'{di["cls"]} {di["score"]:.2f}'); rect(ax[0, 0], dj["box"], "#F1C40F", f'{dj["cls"]} {dj["score"]:.2f}')
    ax[0, 1].imshow(img); ax[0, 1].set_title(r"(b) Union region $\Omega_{ij}$", weight="bold")
    ax[0, 1].add_patch(Rectangle((U[0], U[1]), U[2]-U[0], U[3]-U[1], fc="#C000C0", alpha=0.28, ec="#C000C0", lw=2))
    for a, s, t in [(ax[0, 2], sal_i, di["cls"]), (ax[1, 0], sal_j, dj["cls"])]:
        a.imshow(img); a.imshow(s, cmap="jet", alpha=0.45); a.set_title(f"D-RISE: {t}", weight="bold")
    ax[0, 2].set_title(f'(c) D-RISE: {di["cls"]}', weight="bold"); ax[1, 0].set_title(f'(d) D-RISE: {dj["cls"]}', weight="bold")
    ax[1, 1].imshow(img); ax[1, 1].imshow(np.ma.masked_where(~P["masks"][top], P["masks"][top]), cmap="autumn", alpha=0.6)
    rect(ax[1, 1], U, "white", ls="--"); ax[1, 1].set_title(f"(e) PIE: top-ranked unit U{top+1}", weight="bold")
    b = ax[1, 2]; vals = P["ce"][order][::-1]
    b.barh(range(len(order)), vals, color=["#C0392B" if i == len(order)-1 else "#2E6DA4" for i in range(len(order))], height=0.65)
    b.set_yticks(range(len(order))); b.set_yticklabels([f"U{q+1}" for q in order][::-1]); b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("Confidence effect $CE_k$"); b.set_title("(f) PIE: ranking by $CE_k$", weight="bold"); b.grid(axis="x", alpha=0.3)
    for s in ("top", "right"): b.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig18_person_laptop.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig18_person_laptop.png", dpi=300, bbox_inches="tight"); plt.show()
    S = di["score"] * dj["score"]
    cap = (r"\caption{Person--laptop pair (MS-COCO image " + str(row["image_id"]) + r"), selected as the pair whose "
           r"\ours{} MoRF-AUC is closest to the median of the person--laptop pairs. (a) Detected pair and (b) union region "
           r"$\Omega_{ij}$. (c)--(d) D-RISE produces one map per object. (e) Top-ranked latent unit of \ours{} and (f) all "
           f"$K={K}$" + r" units ranked by $CE_k$; the top unit lowers the joint confidence by " + f"{P['ce'].max():.3f}"
           + r" of $S_{ij}=" + f"{S:.3f}" + r"$.}")
    (OUT / "fig18_caption.tex").write_text(cap); print(cap)

# ------------------------------------------------------------------------- MAIN
t0 = time.time()
img_dir, ann = ensure_coco()
coco = COCO(str(ann)); det = Detector(); rng = np.random.default_rng(SEED)

if RUN_SELECTION:   # Table 6 on train2017 pairs only - never on evaluation pairs
    coco_tr = COCO(str(COCO_ROOT / "annotations/instances_train2017.json"))
    sel_pairs = build_pairs(coco_tr, SELECTION_PAIRS // len(CLASS_PAIRS), rng)
    grid = [("K", v, dict(k=v)) for v in (4, 8, 12)] + [("sigma", v, dict(sigma=v)) for v in (10, 15, 20)] \
         + [("percentile", v, dict(pct=v)) for v in (90, 85, 80)]
    rows = []
    for name, v, kw in grid:
        vals = []
        for p in tqdm(sel_pairs, desc=f"selection {name}={v}"):
            img = np.asarray(Image.open(train_image(p["file_name"])).convert("RGB"))
            r, _ = evaluate_pair(det, img, p, rng, with_drise=False, **kw)
            if r: vals.append(r["morf_PIE"])
        rows.append(dict(parameter=name, value=v, morf_auc=np.mean(vals), n=len(vals)))
    sel = pd.DataFrame(rows); sel.to_csv(OUT / "table6_selection.csv", index=False); print(sel)

pairs = build_pairs(coco, MAX_EVAL_PAIRS_PER_CLASS, rng)
results, skipped, keep = [], {}, {}
for row in tqdm(pairs, desc="evaluation"):
    img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
    try:
        r, extra = evaluate_pair(det, img, row, rng)
    except Exception as e:
        r, extra = None, f"error: {type(e).__name__}"
    if r is None:
        skipped[extra] = skipped.get(extra, 0) + 1; continue
    results.append(r)
    if r["class_pair"] == "person+laptop":
        keep[row["image_id"]] = row
df = pd.DataFrame(results); df.to_csv(OUT / "per_pair_results.csv", index=False)
print(f"\nEvaluated {len(df)} / {len(pairs)} pairs. Skipped (reported, not hidden): {skipped}")
print("Detector passes:", det.n_passes, f"| time {(time.time()-t0)/60:.1f} min\n")
report(df)

# Fig. 18: fixed, pre-declared rule -> person-laptop pair with PIE MoRF closest to the median
pl = df[df.class_pair == "person+laptop"]
fig_id = FIG18_IMAGE_ID or int(pl.iloc[(pl["morf_PIE"] - pl["morf_PIE"].median()).abs().argsort().iloc[0]]["image_id"])
row = keep.get(fig_id) or next(p for p in build_pairs(coco, None, rng) if p["image_id"] == fig_id and p["class_j"] == "laptop")
img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
_, (P, di, dj) = evaluate_pair(det, img, row, rng, with_drise=False)
make_fig18(det, img, row, P, di, dj, rng)

json.dump(dict(model=MODEL_NAME, input=INPUT_SIZE, conf=CONF, nms=NMS_IOU, K=K, tau_s=TAU_S, percentile=PCT,
               sigma=SIGMA, theta_m=THETA_M, fractions=len(FRACTIONS), drise_masks=DRISE_MASKS, seed=SEED,
               pairs_evaluated=len(df), skipped=skipped, fig18_image=fig_id),
          open(OUT / "run_config.json", "w"), indent=2)
import shutil; shutil.make_archive("/content/pie_eval_results", "zip", OUT)
from google.colab import files; files.download("/content/pie_eval_results.zip")
