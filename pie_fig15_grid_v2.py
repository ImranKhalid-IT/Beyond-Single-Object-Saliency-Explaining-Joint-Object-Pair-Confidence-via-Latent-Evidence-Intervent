# ==========================================================================================
# PIE - Fig. 15 v2 (qualitative MS-COCO examples, 2 x 4 grid, bold DejaVu Serif), one Colab cell, GPU runtime. YOLOv5x, val2017.
#
# Every number in the figure is computed by the same PIE pipeline as the evaluation code
# (L2-normalised features, K-Means, softmax maps, top-15% masks, Gaussian blur, same-class
# IoU matching, CE_k = S_ij - s'_i s'_j). Nothing is drawn by hand.
#
# How the examples are chosen (fixed rule, stated in the caption):
#   for every class pair, CANDIDATES_PER_PAIR images are scored and the one with the largest
#   normalised top-unit effect CE_max / S_ij is shown, among candidates whose top unit lowers
#   BOTH detections (shared evidence). SELECTION = "median" shows a typical example instead.
#
# Output (/content/pie_fig15.zip):
#   fig15_qualitative.pdf / .png, fig15_caption.tex,
#   fig15_unit_effects.csv  (CE_k of every unit of every shown pair),
#   fig15_all_candidates.csv (summary of every scored candidate, for transparency)
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
from tqdm.auto import tqdm
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ------------------------------------------------------------------------- CONFIG
# (class_i, class_j, number of different images to show) -> 8 panels in a 2 x 4 grid
EXAMPLES = [("dog", "frisbee", 2), ("person", "laptop", 2), ("person", "horse", 2),
            ("person", "motorcycle", 1), ("person", "cup", 1)]
N_COLS = 4
CANDIDATES_PER_PAIR = 60               # images scored per class pair
SELECTION = "clear"                    # "clear" (largest CE_max/S_ij) or "median" (typical)
MIN_SCORE = 0.50                       # both target detections must be at least this confident
MIN_UNION_FRAC, MAX_UNION_FRAC = 0.04, 0.70   # union box must be visible but not the whole image
MIN_CONTACT = 0.30        # boxes must overlap: intersection / smaller-box area (person on horse, laptop on lap)
MIN_OBJ_AREA = 900        # ignore tiny instances (pixels^2)
MIN_SHARED = 0.20         # top-unit mask must lie at least 20 % on each of the two boxes
MODEL_NAME, INPUT_SIZE = "yolov5x", 640
CONF, NMS_IOU, MAX_DET = 0.25, 0.50, 100
K, TAU_S, PCT, SIGMA, THETA_M = 8, 0.05, 85, 15, 0.50
BATCH, SEED = 32, 42
OUT = Path("/content/pie_fig15"); OUT.mkdir(parents=True, exist_ok=True)
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

def contact(a, b):
    """Overlap of two xyxy boxes relative to the smaller box (1 = one box inside the other)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1]))
    return iw * ih / (small + 1e-8)

def candidates(coco, a, b, rng, n):
    """Interacting pairs only: in every image, the (class_i, class_j) instances whose boxes
    touch the most are taken (not simply the largest ones); pairs with contact below
    MIN_CONTACT are dropped and the rest are ordered by contact."""
    ca, cb = coco.getCatIds(catNms=[a])[0], coco.getCatIds(catNms=[b])[0]
    ids = sorted(set(coco.getImgIds(catIds=[ca])) & set(coco.getImgIds(catIds=[cb])))
    xy = lambda bb: [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
    rows = []
    for iid in ids:
        A = [x for x in coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[ca], iscrowd=False)) if x["area"] >= MIN_OBJ_AREA]
        B = [x for x in coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cb], iscrowd=False)) if x["area"] >= MIN_OBJ_AREA]
        if not A or not B:
            continue
        c, ga, gb = max(((contact(xy(p["bbox"]), xy(q["bbox"])), p, q) for p in A for q in B),
                        key=lambda t: (t[0], t[1]["area"] + t[2]["area"]))
        if c < MIN_CONTACT:
            continue
        rows.append(dict(image_id=iid, file_name=coco.loadImgs([iid])[0]["file_name"], class_i=a, class_j=b,
                         gt_i=xy(ga["bbox"]), gt_j=xy(gb["bbox"]), contact=round(float(c), 3)))
    rng.shuffle(rows)                                   # random order among equal contact
    rows.sort(key=lambda r: -r["contact"])
    print(f"{a}+{b}: {len(rows)} interacting pairs (contact >= {MIN_CONTACT})")
    return rows[:n] if n else rows

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

    def detect_many(self, imgs):
        out = []
        for s in range(0, len(imgs), BATCH):
            with torch.no_grad():
                res = self.m(list(imgs[s:s + BATCH]), size=INPUT_SIZE).xyxy
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

    def features(self, img):
        x, g = self._letterbox(img); box = {}
        def hook(mod, inp):
            maps = [t for t in inp[0] if torch.is_tensor(t) and t.ndim == 4]
            box["f"] = min(maps, key=lambda t: abs(INPUT_SIZE / t.shape[-1] - 16))   # PANet P4, stride 16
        h = self.detect_layer.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                self.m.model(x)
        finally:
            h.remove()
        return box["f"][0].float().detach(), g

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

# ------------------------------------------------------------------------- PIE (identical to evaluation code)
def top_fraction_mask(m, omega, frac):
    ys, xs = np.nonzero(omega); v = m[ys, xs]
    n = max(1, int(math.ceil(frac * len(v))))
    keep = np.argsort(-v, kind="stable")[:n]
    out = np.zeros_like(omega); out[ys[keep], xs[keep]] = True
    return out

def pie_units(det, img, di, dj):
    F_, g = det.features(img); C, h, w = F_.shape
    Fn = F_ / F_.norm(dim=0, keepdim=True).clamp_min(1e-6)
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
    d2 = torch.cdist(Fn.permute(1, 2, 0).reshape(-1, C), mu) ** 2                 # Eq. 21
    Uk = torch.softmax(-d2 / TAU_S, dim=1).T.reshape(K, h, w).cpu().numpy()       # Eq. 22
    omega = box_mask(U, W, H)
    maps = np.stack([np.where(omega, to_pixels(Uk[q], g), 0) for q in range(K)])
    blur = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(SIGMA)))
    masks = np.stack([top_fraction_mask(maps[q], omega, 1 - PCT / 100) for q in range(K)])  # Eqs. 25-26
    post = det.detect_many([np.where(masks[q][..., None], blur, img).astype(np.uint8) for q in range(K)])
    pij = np.array([match_pair(p, di, dj) for p in post])
    ce = di["score"] * dj["score"] - pij[:, 0] * pij[:, 1]                         # Eq. 34
    return dict(maps=maps, masks=masks, omega=omega, U=U, ce=ce, post=pij)

def unit_order(P):
    """Units by CE_k (Eq. 36). Ties (several units at CE_k = S_ij, i.e. a target is lost) are
    broken by the lower summed post-intervention score s'_i + s'_j."""
    return np.lexsort((P["post"].sum(1), -np.round(P["ce"], 6)))

def shared_coverage(mask, bi, bj):
    """Fraction of the top-unit mask lying on box i and on box j; min() of the two."""
    n = mask.sum() + 1e-8
    H, W = mask.shape
    return float(min((mask & box_mask(bi, W, H)).sum() / n, (mask & box_mask(bj, W, H)).sum() / n))

# ------------------------------------------------------------------------- scoring one candidate
def score_candidate(det, img_dir, row):
    img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
    dets = det.detect(img)
    di = pick_detection(dets, row["class_i"], row["gt_i"]); dj = pick_detection(dets, row["class_j"], row["gt_j"])
    base = dict(image_id=row["image_id"], class_pair=f'{row["class_i"]}+{row["class_j"]}', contact=row["contact"])
    if di is None or dj is None:
        return dict(base, status="target not detected")
    if min(di["score"], dj["score"]) < MIN_SCORE:
        return dict(base, status="low confidence")
    H, W = img.shape[:2]; U = union_box(di["box"], dj["box"])
    frac = float((U[2] - U[0]) * (U[3] - U[1]) / (H * W))
    if not (MIN_UNION_FRAC <= frac <= MAX_UNION_FRAC):
        return dict(base, status="union size outside range")
    P = pie_units(det, img, di, dj)
    if P is None:
        return dict(base, status="union too small for K units")
    S = di["score"] * dj["score"]; order = unit_order(P); top, second = int(order[0]), int(order[1])
    both = bool(P["post"][top, 0] < di["score"] and P["post"][top, 1] < dj["score"])
    return dict(base, status="ok", s_i=di["score"], s_j=dj["score"], S_ij=S, ce_max=float(P["ce"][top]),
                ce_norm=float(P["ce"][top] / S), gap_norm=float((P["ce"][top] - P["ce"][second]) / S),
                shared=shared_coverage(P["masks"][top], di["box"], dj["box"]),
                top_lowers_both=both, union_frac=frac)

def select(summ, n):
    """n different images per class pair, by the fixed rule stated in the caption."""
    ok = [s for s in summ if s["status"] == "ok"]
    if not ok:
        return []
    if SELECTION == "clear":
        pool = [s for s in ok if s["top_lowers_both"] and s["shared"] >= MIN_SHARED]
        if len(pool) < n:                               # relax only if too few candidates
            pool = [s for s in ok if s["top_lowers_both"]] or ok
        return sorted(pool, key=lambda s: (-round(s["ce_norm"], 2), -s["gap_norm"], -s["shared"]))[:n]
    med = np.median([s["ce_norm"] for s in ok])
    return sorted(ok, key=lambda s: abs(s["ce_norm"] - med))[:n]

# ------------------------------------------------------------------------- drawing
def crop_window(U, W, H, pad=0.30, aspect=4 / 3):
    cx, cy = (U[0] + U[2]) / 2, (U[1] + U[3]) / 2
    w, h = (U[2] - U[0]) * (1 + 2 * pad), (U[3] - U[1]) * (1 + 2 * pad)
    if w / h < aspect: w = h * aspect
    else: h = w / aspect
    w, h = min(w, W), min(h, H)
    return float(np.clip(cx - w / 2, 0, W - w)), float(np.clip(cy - h / 2, 0, H - h)), w, h

def img_panel(ax, img, win, title):
    ax.imshow(img, interpolation="none"); x0, y0, w, h = win
    ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=5.6, pad=2)
    for s in ax.spines.values(): s.set_linewidth(0.5)

def draw_example(fig, cell, num, row, img, di, dj, P):
    """One example block: 1 detected pair | 2 union region / 3 top unit | 4 ranking."""
    U, ce, S = P["U"], P["ce"], di["score"] * dj["score"]
    H, W = img.shape[:2]; win = crop_window(U, W, H)
    order = unit_order(P); top = int(order[0])
    hdr = fig.add_subplot(cell); hdr.axis("off")
    hdr.set_title(f'({num}) {row["class_i"]} + {row["class_j"]}', fontsize=7, weight="bold", pad=10)
    g = cell.subgridspec(2, 2, wspace=0.34, hspace=0.32, width_ratios=[1, 1])

    a = fig.add_subplot(g[0, 0]); img_panel(a, img, win, "1. Pair")
    a.text(0.02, 0.03, f'COCO {row["image_id"]}', transform=a.transAxes, fontsize=4.3, color="white",
           va="bottom", bbox=dict(fc="black", alpha=0.55, ec="none", pad=0.5))
    for d, col in ((di, "#17BECF"), (dj, "#F1C40F")):
        b = d["box"]
        a.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=col, lw=1.1))
        a.text(b[0], b[1], f'{d["cls"]} {d["score"]:.2f}', fontsize=4.8, va="bottom", weight="bold",
               bbox=dict(fc=col, ec="none", pad=0.6), clip_on=True)

    a = fig.add_subplot(g[0, 1]); img_panel(a, img, win, r"2. Union $\Omega_{ij}$")
    a.add_patch(Rectangle((U[0], U[1]), U[2]-U[0], U[3]-U[1], fc="#C000C0", alpha=0.28, ec="#C000C0", lw=1.1))

    a = fig.add_subplot(g[1, 0]); img_panel(a, img, win, f"3. Top: U{top+1}")
    m = P["masks"][top]; heat = P["maps"][top].copy()
    v = heat[m]; heat = (heat - v.min()) / (v.max() - v.min() + 1e-8)
    a.imshow(np.where(m, heat, np.nan), cmap="jet", alpha=0.60, vmin=0, vmax=1, interpolation="none")
    a.add_patch(Rectangle((U[0], U[1]), U[2]-U[0], U[3]-U[1], fill=False, ec="white", lw=0.9, ls="--"))

    a = fig.add_subplot(g[1, 1]); vals = ce[order]; y = np.arange(K)[::-1]
    cols = ["#E8743B"] + ["#F2A97E" if v >= 0 else "#B0B0B0" for v in vals[1:]]
    a.barh(y, vals, color=cols, height=0.7, edgecolor="#7a3b1b", linewidth=0.3)
    for yy, vv in zip(y, vals):
        a.text(max(vv, 0) + 0.02 * S, yy, f"{vv:.3f}", fontsize=4.3, va="center", ha="left")
    a.set_yticks(y); a.set_yticklabels([f"U{q+1}" for q in order], fontsize=4.8)
    a.axvline(0, color="k", lw=0.4); a.axvline(S, color="k", lw=0.5, ls="--")
    a.text(S, -0.95, r"$S_{ij}$", fontsize=4.8, ha="center", va="top")
    a.set_xlim(min(0.0, vals.min()) - 0.03 * S, S * 1.38); a.set_ylim(-0.6, K - 0.4)
    a.tick_params(axis="x", labelsize=4.5, length=1.5, pad=1); a.tick_params(axis="y", length=0, pad=1)
    a.set_xlabel(r"$CE_k$", fontsize=5, labelpad=3)
    a.set_title("4. $CE_k$ ranking", fontsize=5.6, pad=2)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): a.spines[s_].set_linewidth(0.4)

# ------------------------------------------------------------------------- MAIN
t0 = time.time()
img_dir, ann = ensure_coco(); coco = COCO(str(ann)); det = Detector(); rng = np.random.default_rng(SEED)
all_summ, chosen = [], []
for a, b, n_show in EXAMPLES:
    rows = candidates(coco, a, b, rng, CANDIDATES_PER_PAIR)
    summ = [score_candidate(det, img_dir, r) for r in tqdm(rows, desc=f"{a}+{b}")]
    all_summ += summ
    picks = select(summ, n_show)
    if len(picks) < n_show:
        print(f"only {len(picks)} valid candidate(s) for {a}+{b}")
    for pk in picks:
        chosen.append(next(r for r in rows if r["image_id"] == pk["image_id"]))
pd.DataFrame(all_summ).to_csv(OUT / "fig15_all_candidates.csv", index=False)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
                     "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold",
                     "mathtext.fontset": "dejavuserif", "mathtext.default": "bf",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
n_rows = math.ceil(len(chosen) / N_COLS)
fig = plt.figure(figsize=(7.16, 2.05 * n_rows))
outer = fig.add_gridspec(n_rows, N_COLS, wspace=0.10, hspace=0.30)
unit_rows = []
for n, row in enumerate(chosen):                     # recompute the chosen pairs (deterministic)
    img = np.asarray(Image.open(img_dir / row["file_name"]).convert("RGB"))
    dets = det.detect(img)
    di, dj = pick_detection(dets, row["class_i"], row["gt_i"]), pick_detection(dets, row["class_j"], row["gt_j"])
    P = pie_units(det, img, di, dj)
    draw_example(fig, outer[n // N_COLS, n % N_COLS], n + 1, row, img, di, dj, P)
    for q in range(K):
        unit_rows.append(dict(image_id=row["image_id"], class_pair=f'{row["class_i"]}+{row["class_j"]}',
                              unit=f"U{q+1}", CE_k=P["ce"][q], post_s_i=P["post"][q, 0], post_s_j=P["post"][q, 1],
                              s_i=di["score"], s_j=dj["score"], S_ij=di["score"] * dj["score"]))
fig.savefig(OUT / "fig15_qualitative.pdf", bbox_inches="tight", dpi=600)
fig.savefig(OUT / "fig15_qualitative.png", dpi=600, bbox_inches="tight"); plt.show()
ue = pd.DataFrame(unit_rows); ue.to_csv(OUT / "fig15_unit_effects.csv", index=False)

rule = (r"Only pairs whose boxes overlap (intersection over the smaller box $\geq$ " + str(MIN_CONTACT) + r") are considered. For each class pair, the examples are the candidates with the largest normalized top-unit effect $CE_{\max}/S_{ij}$ among "
        + str(CANDIDATES_PER_PAIR) + r" scored images whose top unit lowers both detections and lies at least " + str(int(100 * MIN_SHARED)) + r"\% on each box"
        if SELECTION == "clear" else
        r"Only pairs whose boxes overlap (intersection over the smaller box $\geq$ " + str(MIN_CONTACT) + r") are considered. Each example is the candidate whose normalized top-unit effect $CE_{\max}/S_{ij}$ is closest to the median of "
        + str(CANDIDATES_PER_PAIR) + r" scored images of that class pair")
cap = (r"\caption{MS-COCO examples of \ours{} with YOLOv5. For each pair, the panels show the detected pair, the union "
       r"region $\Omega_{ij}$, the top-ranked latent unit, and the confidence effects $CE_k$ of all $K=8$ units "
       r"(dashed line: $S_{ij}$; gray bars: negative effects; units tied at $CE_k=S_{ij}$, where a target is lost, are ordered by the lower remaining score $s'_i+s'_j$). " + rule + r". Unit indices carry no meaning across images.}")
(OUT / "fig15_caption.tex").write_text(cap); print("\n" + cap)
print("\nShown pairs:")
print(ue.groupby(["class_pair", "image_id"]).agg(S_ij=("S_ij", "first"), CE_max=("CE_k", "max"),
      n_negative=("CE_k", lambda x: int((x < 0).sum()))).round(3).to_string())

json.dump(dict(model=MODEL_NAME, K=K, tau_s=TAU_S, percentile=PCT, sigma=SIGMA, theta_m=THETA_M,
               selection=SELECTION, candidates_per_pair=CANDIDATES_PER_PAIR, min_score=MIN_SCORE,
               union_frac=[MIN_UNION_FRAC, MAX_UNION_FRAC], seed=SEED, minutes=(time.time() - t0) / 60),
          open(OUT / "run_config.json", "w"), indent=2)
shutil.make_archive("/content/pie_fig15", "zip", OUT)
from google.colab import files; files.download("/content/pie_fig15.zip")
