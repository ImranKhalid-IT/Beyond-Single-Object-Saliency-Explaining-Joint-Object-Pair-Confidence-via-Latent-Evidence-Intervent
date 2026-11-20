# PIE: Pair-level Interventional Explanation

**Beyond Single-Object Saliency: Explaining Joint Object-Pair Confidence via Latent Evidence Interventions**



## Overview

Object detectors are usually explained one detection at a time, yet many downstream decisions depend on **two detections together**. PIE is a post-hoc framework that explains the **joint confidence** of a selected detection pair in a **frozen** detector:

$$S_{ij} = s_i \, s_j$$

When an intervention changes the two scores by $\Delta s_i$ and $\Delta s_j$, the joint confidence changes by

$$\Delta S_{ij} = s_j\,\Delta s_i + s_i\,\Delta s_j + \Delta s_i\,\Delta s_j .$$

The cross term $\Delta s_i\Delta s_j$ is non-zero only when the same evidence affects both detections. Merged object-wise maps and first-order (gradient) attributions cannot represent it. PIE measures $\Delta S_{ij}$ directly, so the cross term is included by construction.

**Key properties**

- Explains the joint confidence of a detection pair, not a single score.
- Ranks image-specific latent evidence units by their **measured** drop in $S_{ij}$, not by activation or gradient.
- Pair-preserving one-to-one IoU matching keeps the explanation defined when detections shift or disappear.
- Works with one-stage (YOLOv5), two-stage (Faster R-CNN) and transformer-based (DETR) detectors through readout adapters, **without retraining**.
- Needs only $K+1$ forward passes per pair (9 passes with $K=8$), under one second per pair.

> **Scope.** PIE explains detector confidence for object pairs. It does **not** predict, explain, or verify predicates or semantic relationships.

---

## Method

PIE has four stages:

1. **Target-pair definition.** Select two detections $d_i, d_j$ and compute their union box $b_{ij}$ and pixel region $\Omega_{ij}$. All interventions stay inside $\Omega_{ij}$.
2. **Detector-family feature readout.** Read features from a fixed layer of the frozen detector inside the union region:
   | Detector family | Model | Readout | Layer |
   |---|---|---|---|
   | One-stage | YOLOv5x | Grid sampling | PANet neck P4 (stride 16) |
   | Two-stage | Faster R-CNN R101-FPN | ROI-Align on $b_{ij}$ ($7\times7$, sampling ratio 2) | FPN level P2–P5 chosen by the standard level-assignment rule |
   | Transformer | DETR-DC5 R50 | Encoder tokens inside $\Omega_{ij}$ | 6th encoder layer |
3. **Latent evidence units.** K-Means groups the located features into $K$ prototypes; a temperature softmax gives unit maps $U_k$ that are resampled to pixel maps $\widetilde{U}_k$ inside $\Omega_{ij}$.
4. **Interventional ranking.** Each map is thresholded at its 85th percentile to give a mask $M_k$ (strongest 15%). The masked pixels are replaced with a Gaussian-blurred copy of the image, the detector is re-run, the two targets are recovered by same-class one-to-one IoU matching ($\theta_m = 0.5$), and each unit is scored by

   $$CE_k = S_{ij}(I) - \widehat{s}'_{i,k}\,\widehat{s}'_{j,k}.$$

   An unmatched target is treated as lost (score 0). Units are sorted by $CE_k$ (negative effects are kept, not clipped).

For display only, the top-ranked mask can be refined with DeepLabV3. This refinement does **not** affect $CE_k$.

---

## Default configuration

All detectors and both datasets use the same settings (paper, Table V):

| Parameter | Symbol | Value |
|---|---|---|
| Number of latent units | $K$ | 8 |
| Softmax temperature | $\tau_s$ | 0.05 |
| Mask percentile | – | 85 (keeps strongest 15%) |
| Gaussian blur scale | $\sigma$ | 15 |
| Matching IoU threshold | $\theta_m$ | 0.50 |
| ROI-Align grid (Faster R-CNN only) | $G$ | $7\times7$ |
| Detector confidence threshold | – | 0.25 |
| NMS IoU threshold | – | 0.50 |
| Max detections per image | – | 100 |

---

## Repository contents

| File | Description |
|---|---|
| `pie_full_eval_colab.py` | Full PIE evaluation pipeline (pair-level MoRF deletion, insertion, ablations) — runs on Google Colab or a local GPU |
| `pie_fig15_grid_v2.py` | Builds the multi-example qualitative figure (selected pair, union region, top-ranked unit, $CE_k$ ranking) |
| `demo_rce.py` | Single-image demo of PIE on one detection pair |
| `comperision.py` | Qualitative comparison of PIE with D-RISE, ODAM, Finer-CAM and CRAFT on the same image and pair |
| `Testing Bench Mark.ipynb` | Notebook for the benchmark runs |
| `pie-xai.zip`, `RCE.zip` | Packaged source code |
| `RCE_Non intrection experiments.zip` | Visual Genome predicate-annotated vs. non-annotated pair experiment |
| `Table_VI_Faithfulness.csv` | MoRF-AUC results (paper Table VI) |
| `Table_VII_Efficiency.csv` | Explanation time per pair (paper Table X) |
| `*.png` | Example outputs and figures |

> **Note:** "RCE" is the earlier working name of the method. It refers to the same method as **PIE** in the paper.

---

## Installation

Requirements: Python 3.10+, PyTorch 2.2+, torchvision 0.17+, a CUDA GPU (recommended).

```bash
git clone https://github.com/ImranKhalid-IT/Beyond-Single-Object-Saliency-Explaining-Joint-Object-Pair-Confidence-via-Latent-Evidence-Intervent.git
cd Beyond-Single-Object-Saliency-Explaining-Joint-Object-Pair-Confidence-via-Latent-Evidence-Intervent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision numpy pillow matplotlib scikit-learn opencv-python scipy tqdm pycocotools
pip install ultralytics            # YOLOv5
pip install 'git+https://github.com/facebookresearch/detectron2.git'   # Faster R-CNN
```

### Detector checkpoints (public COCO weights)

| Detector | Checkpoint | Source | COCO mAP |
|---|---|---|---|
| YOLOv5x | `yolov5x.pt` | Ultralytics | 50.7 |
| Faster R-CNN R101-FPN 3x | `faster_rcnn_R_101_FPN_3x` (`model_final_f6e8b1.pkl`) | Detectron2 Model Zoo | 42.0 |
| DETR-DC5 R50 | `detr-r50-dc5-f0fb7ef5.pth` | Official DETR repository | 43.3 |

All detectors are kept frozen.

---

## Datasets

| Dataset | Use | Download |
|---|---|---|
| MS-COCO 2017 validation | Main benchmark: 4,955 co-occurring detection pairs | https://cocodataset.org/ |
| Visual Genome | 1,200 pairs: 600 with an annotated predicate, 600 co-occurring pairs without one (box IoU < 0.1) | https://visualgenome.org/ |

MS-COCO class pairs used in the benchmark:

| Class pair | Pairs |
|---|---|
| person + bicycle | 1,520 |
| person + laptop | 1,285 |
| car + traffic light | 2,150 |
| **Total** | **4,955** |

Visual Genome predicate annotations are used **only** to stratify pairs. PIE does not predict or explain predicates.

Suggested layout:

```
data/
├── coco/
│   ├── val2017/
│   └── annotations/instances_val2017.json
└── visual_genome/
    ├── images/
    └── relationships.json
```

---

## Usage

### Single-pair demo

```bash
python demo_rce.py --image Test_Image.png
```

The demo detects objects, selects a target pair, computes the union region, discovers $K$ latent units, runs the $K$ interventions, and saves the unit maps, the masks and the $CE_k$ ranking.

### Full evaluation

```bash
python pie_full_eval_colab.py
```

This reproduces the pair-level MoRF-AUC and insertion results, the ablation and spatial controls, the hyperparameter sweeps, and the replacement-operator robustness study.

### Qualitative figures

```bash
python pie_fig15_grid_v2.py     # multi-example qualitative figure
python comperision.py           # PIE vs. baselines on one pair
```

> Check each script's header or `--help` for the exact arguments and data paths.

---

## Evaluation protocol

**Pair-level MoRF deletion.** Every method gives a ranking of the pixels in $\Omega_{ij}$. For PIE, pixels covered by a mask take the rank of the highest-ranked covering unit; other pixels take the rank of their dominant unit. At each fraction $\alpha$, the top $\lfloor\alpha|\Omega_{ij}|\rfloor$ pixels are replaced with the **same** Gaussian-blurred image for all methods, the detector is re-run, the pair is matched, and

$$\mathrm{MoRF\text{-}AUC} = \int_0^1 \frac{S(\alpha)}{S(0)}\,d\alpha \qquad (\text{lower is better}).$$

**Insertion AUC.** $\Omega_{ij}$ starts fully blurred and pixels are restored in ranking order (higher is better).

**Baselines.** D-RISE, ODAM and Finer-CAM explain $d_i$ and $d_j$ separately, and their normalized maps are averaged. GCE and CRAFT are restricted to $\Omega_{ij}$, and CoE features are aligned with $\Omega_{ij}$. Finer-CAM was not evaluated on DETR.

---

## Main results

### Faithfulness on 4,955 MS-COCO pairs (MoRF-AUC ↓)

| Method | YOLOv5 | Faster R-CNN | DETR | Mean |
|---|---|---|---|---|
| Finer-CAM | 0.51 | 0.48 | – | 0.50* |
| CRAFT | 0.48 | 0.46 | 0.52 | 0.49 |
| D-RISE | 0.45 | 0.42 | 0.49 | 0.45 |
| GCE | 0.43 | 0.45 | 0.47 | 0.45 |
| CoE | 0.41 | 0.42 | 0.43 | 0.42 |
| ODAM | 0.38 | 0.40 | 0.42 | 0.40 |
| **PIE (ours)** | **0.25** | **0.23** | **0.27** | **0.25** |

\*Mean over the two evaluated detectors.

### Effect-based vs. first-order ranking of the same units

| Detector | PIE MoRF ↓ | First-order MoRF ↓ | PIE Insertion ↑ | First-order Insertion ↑ |
|---|---|---|---|---|
| YOLOv5 | **0.25** | 0.28 | **0.72** | 0.68 |
| Faster R-CNN | **0.23** | 0.26 | **0.74** | 0.70 |
| DETR | **0.27** | 0.30 | **0.70** | 0.66 |
| **Mean** | **0.25** | 0.28 | **0.72** | 0.68 |

### Ablation and spatial controls (mean MoRF-AUC ↓)

| Variant | MoRF-AUC |
|---|---|
| Detector baseline | 0.45 |
| Random spatial masking | 0.41 |
| Pair-union attribution | 0.38 |
| PC-Grid ($8\times8$ cells, $CE$ ranking) | 0.36 |
| Latent units, activation-ranked | 0.31 |
| Latent units, first-order-ranked | 0.28 |
| **PIE (latent units, $CE_k$-ranked)** | **0.25** |

The 0.06 gain of effect-based over activation-based ranking splits into 0.03 from the first-order terms and 0.03 from the cross term.

### Robustness to the replacement operator

| Replacement | MoRF-AUC ↓ | Spearman ρ ↑ | Latency (s) |
|---|---|---|---|
| Gaussian blur (default) | 0.250 | 1.00 | 0.82 |
| Mean replacement | 0.263 | 0.91 | 0.84 |
| Uniform noise | 0.265 | 0.89 | 0.83 |
| Structure-preserving inpainting | 0.271 | 0.87 | 1.05 |

### Explanation time per pair (seconds)

| Method | YOLOv5 | Faster R-CNN | DETR |
|---|---|---|---|
| D-RISE | 15.21 | 18.45 | 22.10 |
| CoE | 1.25 | 1.42 | 1.65 |
| CRAFT | 0.82 | 0.90 | 0.98 |
| ODAM | 0.35 | 0.42 | 0.50 |
| **PIE (ours)** | **0.75** | **0.82** | **0.95** |

### Visual Genome (1,200 pairs)

- MoRF-AUC: **PIE 0.28** vs. ODAM 0.43.
- Largest effect per pair, $CE_{\max}$: 0.214 (s.d. 0.082) for predicate-annotated pairs and 0.083 (s.d. 0.041) for non-annotated pairs (IoU < 0.1).

This comparison is descriptive: initial confidence and spatial separation also differ between the two groups, so it does not show that PIE detects relationships.

---

## Limitations

- **Scope:** PIE explains $S_{ij}$; it does not predict, explain, or verify predicates.
- **Coverage:** three MS-COCO class pairs, a 1,200-pair Visual Genome subset, and one detector per family. Other detectors need an analogous readout adapter.
- **Protocol:** baselines are adapted to the pair setting; the metric shares PIE's replacement operator; $K$, $\sigma$ and the mask percentile were selected on the evaluation pairs.
- **Units:** units are image-specific prototypes; unit names shown in figures are manually assigned display tags. All replacement operators are synthetic, so $CE_k$ measures the model's response to an intervention, not to a physical scene change.
- **Cost:** $K+1$ forward passes per pair; all pairs among $n$ detections need $n(n-1)/2$ explanations, so PIE targets selected pairs rather than exhaustive analysis of crowded scenes.

---

## Citation

If you use this code, please cite:

```bibtex
@article{khalid2026pie,
  title   = {Beyond Single-Object Saliency: Explaining Joint Object-Pair Confidence via Latent Evidence Interventions},
  author  = {Khalid, Muhammad Imran},
  journal = {IEEE Access},
  note    = {Under review},
  year    = {2026}
}
```

Related work by the author:

```bibtex
@article{khalid2026interpretable,
  title   = {Interpretable Object Detection via Integrated Heatmap, Concept Attribution, and {Sobol} Sensitivity Analysis},
  author  = {Khalid, Muhammad Imran and Mi, Jian-Xun and Ali, G and Ali, Tariq and Hijji, M and Ayaz, M},
  journal = {Information Sciences},
  pages   = {123133},
  year    = {2026}
}

@inproceedings{khalid2025global,
  title     = {Global Concept-Based Explanations ({GCE}): A Universal Concept Dictionary for Interpretable Object Detection},
  author    = {Khalid, M I and Mi, J-X and Komal, A and Sarwar, N and Attique, D},
  booktitle = {2025 IEEE 19th International Conference on Open Source Systems and Technologies (ICOSST)},
  pages     = {1--6},
  year      = {2025},
  doi       = {10.1109/ICOSST69113.2025.11315337}
}

@article{khalid2026decomposing,
  title   = {Decomposing Decisions: Causal Concept Explanations for Deep Learning Models},
  author  = {Khalid, Muhammad Imran and Mi, Jian-Xun},
  journal = {Journal of Experimental \& Theoretical Artificial Intelligence},
  pages   = {1--19},
  year    = {2026},
  doi     = {10.1080/0952813X.2026.2701277}
}
```

---

## Acknowledgment

The author thanks Prof. Jian-Xun Mi for his supervision, guidance, and continuous support throughout this research.

We thank the authors of [YOLOv5](https://github.com/ultralytics/yolov5), [Detectron2](https://github.com/facebookresearch/detectron2), [DETR](https://github.com/facebookresearch/detr), [MS-COCO](https://cocodataset.org/) and [Visual Genome](https://visualgenome.org/) for making their models and data public.

## Contact

For questions, please open an issue in this repository.
