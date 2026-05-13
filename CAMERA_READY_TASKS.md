# TDSal Camera-Ready: Technical Tasks for Claude Code

This file defines the concrete, executable technical tasks required for the CGI 2026 camera-ready revision. All tasks are grounded in the existing notebook code. Run them in order.

---

## Repo Structure (Expected)

```
tdsal/
├── model.py               # YOLOTaskSaliencyModel, all components
├── dataset.py             # TaskSaliencyDataset
├── train.py
├── eval.py
├── checkpoints/
│   └── tdsp_50epoch.pth   # Best trained weights
├── scripts/               # New scripts go here (created by these tasks)
└── paper/
    └── paper0005.tex
```

---

## Task 1 — Efficiency Metrics

**What:** Compute trainable parameter count, FLOPs, and inference time.  
**Why:** All three reviewers flagged the absence of efficiency metrics. These numbers go into Table 1 and Section 3.2.

**Create:** `scripts/compute_efficiency.py`

```python
"""
Compute efficiency metrics for TDSal (YOLOTaskSaliencyModel).

Usage:
    python scripts/compute_efficiency.py --checkpoint checkpoints/tdsp_50epoch.pth
"""

import argparse
import time
import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, images, task_descs, device, n_runs=100, warmup=10):
    model.eval()
    model.to(device)
    images = images.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(images, task_descs)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images, task_descs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    from model import YOLOTaskSaliencyModel

    model = YOLOTaskSaliencyModel()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Dummy inputs matching the notebook's forward signature:
    # model(images, task_descriptions) where task_descriptions is a list of strings
    dummy_images = torch.randn(1, 3, 384, 384)
    dummy_tasks = ["free view"]

    # --- Parameter count ---
    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Trainable parameters: {n_params:,}  ({n_params/1e6:.2f} M)")

    # --- FLOPs ---
    # thop's profile() requires tensor-only inputs.
    # Because task_descriptions goes through SentenceTransformer (not a tensor),
    # we wrap the model and fix the task string for profiling.
    try:
        from thop import profile, clever_format

        class ModelWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x, ["free view"])

        wrapper = ModelWrapper(model)
        flops, _ = profile(wrapper, inputs=(dummy_images,), verbose=False)
        flops_str, _ = clever_format([flops, n_params], "%.2f")
        print(f"FLOPs (single image, visual path): {flops_str}")
    except ImportError:
        print("thop not installed — run: pip install thop")

    # --- Inference time: CPU ---
    cpu = torch.device("cpu")
    mean_cpu, std_cpu = measure_inference_time(model, dummy_images, dummy_tasks, cpu)
    print(f"Inference time (CPU): {mean_cpu:.1f} +/- {std_cpu:.1f} ms  (n=100)")

    # --- Inference time: GPU ---
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        mean_gpu, std_gpu = measure_inference_time(model, dummy_images, dummy_tasks, gpu)
        print(f"Inference time (GPU): {mean_gpu:.1f} +/- {std_gpu:.1f} ms  (n=100)")
    else:
        print("Inference time (GPU): CUDA not available")

    print(f"{'='*50}\n")
    print("-> Add these numbers to Table 1 (Implementation Details) in paper0005.tex")
    print("-> Add a 1-2 sentence efficiency paragraph to Section 3.2")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
pip install thop
python scripts/compute_efficiency.py --checkpoint checkpoints/tdsp_50epoch.pth
```

**What to add to the paper (Section 3.2, after Table 1):**
```latex
TDSal maintains a lightweight computational profile: it comprises
approximately \textbf{X.X~M} trainable parameters, requires
\textbf{X.X~GFLOPs} for a single forward pass on a $384{\times}384$
input, and achieves a mean inference time of \textbf{X~ms} on CPU
(averaged over 100 runs). The YOLO backbone weights are frozen during
training, so only the FPM, TaskEncoder projection layer,
TransformerFusion, and SaliencyDecoder are updated via gradient descent.
```

---

## Task 2 — YOLO Backbone Layer Clarification

**What:** Document exactly which layers of the YOLOv5su backbone are used, what their output shapes are, and how the 512-channel tensor reaches the FPM.  
**Why:** Reviewer 1 asked explicitly how many multiscale YOLO features are fused before the FPM.

**Context from the notebook:**

The backbone is defined as:
```python
self.feature_extractor = self.yolo_model.model.model[:10]
```

From the printed model summary in the notebook, layers 0–9 are:
- `(0)` Conv: 3→32, stride 2, kernel 6×6
- `(1)` Conv: 32→64, stride 2, kernel 3×3
- `(2)` C3: 64→64
- `(3)` Conv: 64→128, stride 2
- `(4)` C3: 128→128
- `(5)` Conv: 128→256, stride 2
- `(6)` C3: 256→256
- `(7)` Conv: 256→512, stride 2
- `(8)` C3: 512→512
- `(9)` SPPF: 512→512 (spatial pyramid pooling with max-pool k=5)

Layer 9 (SPPF) output is a single-scale feature map of shape `[B, 512, H/32, W/32]`. For a 384×384 input that is `[B, 512, 12, 12]`. This is the only feature passed to the FPM — there is no multi-scale fusion before the FPM.

**Create:** `scripts/verify_backbone_shapes.py`

```python
"""
Verify the actual output shape of YOLOBackbone on a 384x384 input.
Confirms the layer count and output channels feeding into SimpleFPN.

Usage:
    python scripts/verify_backbone_shapes.py
"""

import torch
from model import YOLOTaskSaliencyModel

model = YOLOTaskSaliencyModel()
model.eval()

dummy = torch.randn(1, 3, 384, 384)

with torch.no_grad():
    backbone_out = model.backbone(dummy)
    fpn_out = model.fpn(backbone_out)
    print(f"Input shape:        {dummy.shape}")
    print(f"Backbone output:    {backbone_out.shape}")   # expect [1, 512, 12, 12]
    print(f"FPM output:         {fpn_out.shape}")        # expect [1, 128, 12, 12]
    print(f"Number of backbone layers: {len(list(model.backbone.feature_extractor))}")
```

**Run:**
```bash
python scripts/verify_backbone_shapes.py
```

**What to add to the paper (Section 3.2.1), replacing any vague "multiscale" phrasing:**
```latex
The YOLOv5su backbone is truncated at layer~9 (SPPF), retaining only
the feature extraction stack (layers~0--9) and discarding the
detection head entirely. For a $384{\times}384$ input, this yields a
single feature map of shape $[B,\,512,\,12,\,12]$, representing a
spatial stride of~32. This single-scale output is passed directly to
the Feature Projection Module (FPM); no multi-scale pyramid is
constructed prior to the FPM. The FPM applies a $1{\times}1$
convolution to reduce the 512-channel map to 128 channels before
transformer fusion.
```

---

## Task 3 — AUC-Borji Sign Fix and Ablation Recomputation

**What:** The notebook's ablation table prints negative AUC-Borji values (e.g., `-0.9264`). AUC-Borji is bounded `[0, 1]` — the negative values are a sign error in `np.trapz` integration direction. Recompute the ablation table with the fix applied.

**Root cause:**
```python
# Original (wrong): fp decreases as threshold increases, so trapz returns negative
auc = np.trapz(tp, fp)

# Fix: negate to get positive area
auc = -np.trapz(tp, fp)
```

**Create:** `scripts/recompute_ablation.py`

```python
"""
Recompute ablation study metrics with corrected AUC-Borji sign.
Outputs a LaTeX table ready to paste into paper0005.tex.

Usage:
    python scripts/recompute_ablation.py \
        --model_dir /path/to/TDYSN/models \
        --data_root /path/to/dataset
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

EPS = 1e-8

def cc_metric(pred, gt):
    B = pred.shape[0]
    pred = pred.view(B, -1)
    gt = gt.view(B, -1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    gt = gt - gt.mean(dim=1, keepdim=True)
    num = (pred * gt).sum(dim=1)
    den = torch.sqrt((pred**2).sum(dim=1) * (gt**2).sum(dim=1) + EPS)
    return (num / den).mean().item()

def kl_metric(pred, gt):
    B = pred.shape[0]
    pred = pred.view(B, -1).clamp(min=EPS)
    gt = gt.view(B, -1).clamp(min=EPS)
    pred = pred / (pred.sum(dim=1, keepdim=True) + EPS)
    gt = gt / (gt.sum(dim=1, keepdim=True) + EPS)
    return (gt * (gt.log() - pred.log())).sum(dim=1).mean().item()

def sim_metric(pred, gt):
    B = pred.shape[0]
    pred = pred.view(B, -1).clamp(min=0)
    gt = gt.view(B, -1).clamp(min=0)
    pred = pred / (pred.sum(dim=1, keepdim=True) + EPS)
    gt = gt / (gt.sum(dim=1, keepdim=True) + EPS)
    return torch.min(pred, gt).sum(dim=1).mean().item()

def nss_metric(pred, fix):
    B = pred.shape[0]
    s = pred.view(B, -1)
    f = fix.view(B, -1)
    s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + EPS)
    n_fix = f.sum(dim=1).clamp(min=1)
    return ((s * f).sum(dim=1) / n_fix).mean().item()

def auc_borji_metric(pred, fix, n_splits=100, step=0.1):
    """Corrected AUC-Borji: always in [0, 1]."""
    pred = pred.detach().cpu().numpy()
    fix = fix.detach().cpu().numpy()
    B = pred.shape[0]
    aucs = []
    for b in range(B):
        s_map = pred[b, 0]
        f_map = fix[b, 0].astype(bool)
        if f_map.sum() == 0:
            continue
        S = (s_map - s_map.min()) / (s_map.max() - s_map.min() + EPS)
        S_fix = S[f_map]
        neg_idx = np.where(~f_map.flatten())[0]
        if len(neg_idx) == 0:
            continue
        for _ in range(n_splits):
            rand_neg = np.random.choice(
                neg_idx, size=S_fix.size,
                replace=(len(neg_idx) < S_fix.size)
            )
            S_rand = S.flatten()[rand_neg]
            thresholds = np.arange(0, 1 + step, step)
            tp = np.array([(S_fix >= t).mean() for t in thresholds])
            fp = np.array([(S_rand >= t).mean() for t in thresholds])
            auc = -np.trapz(tp, fp)   # FIXED sign
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")

def evaluate_model(model, loader, device):
    model.eval()
    scores = {k: [] for k in ["CC", "KL", "SIM", "NSS", "AUC-Borji"]}
    task_map = {"free view": 0, "count people": 1, "detect the emotion": 2, "identify the action": 3}

    with torch.no_grad():
        for b in loader:
            imgs = b["stimuli"].to(device)
            gt = b["fdm"].to(device)
            tasks = b["task_description"]
            task_ids = None
            if getattr(model, "use_sbert", True) is False:
                task_ids = torch.tensor([task_map[t] for t in tasks], dtype=torch.long)
            pred = model(imgs, task_desc=tasks, task_ids=task_ids)
            pred = F.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=False)
            fix = (gt > 0.5).float()
            scores["CC"].append(cc_metric(pred, gt))
            scores["KL"].append(kl_metric(pred, gt))
            scores["SIM"].append(sim_metric(pred, gt))
            scores["NSS"].append(nss_metric(pred, fix))
            scores["AUC-Borji"].append(auc_borji_metric(pred, fix))

    return {k: float(np.mean(v)) for k, v in scores.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    from model import TDYSN
    from dataset import TaskSaliencyDataset

    task_mapping = {
        "task1": "free view", "task2": "count people",
        "task3": "detect the emotion", "task4": "identify the action",
    }
    transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])
    dataset = TaskSaliencyDataset(args.data_root, task_mapping, transform, transform)

    total = len(dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tasks = 4

    runs = [
        ("Full TDSal",      dict(use_task=True,  use_transformer=True,  use_sbert=True,  use_fpm=True),  "full.pth"),
        ("w/o Task",        dict(use_task=False, use_transformer=True,  use_sbert=True,  use_fpm=True),  "no_task.pth"),
        ("w/o Transformer", dict(use_task=True,  use_transformer=False, use_sbert=True,  use_fpm=True),  "no_transformer.pth"),
        ("w/o SBERT",       dict(use_task=True,  use_transformer=True,  use_sbert=False, use_fpm=True),  "no_sbert.pth"),
        ("w/o FPM",         dict(use_task=True,  use_transformer=True,  use_sbert=True,  use_fpm=False), "no_fpm.pth"),
    ]

    rows = []
    for name, cfg, fname in runs:
        path = os.path.join(args.model_dir, fname)
        model = TDYSN(num_tasks=num_tasks, **cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        res = evaluate_model(model, test_loader, device)
        res["Model"] = name
        rows.append(res)
        print(f"Done: {name} -> {res}")

    df = pd.DataFrame(rows)[["Model", "CC", "KL", "SIM", "NSS", "AUC-Borji"]].round(4)
    print("\n", df.to_string(index=False))
    print("\nLaTeX:\n")
    print(df.to_latex(index=False, caption="Ablation study on TDSal components (test set, $n=296$).",
                      label="tab:ablation", column_format="lccccc"))

if __name__ == "__main__":
    main()
```

**Run:**
```bash
pip install pandas
python scripts/recompute_ablation.py \
    --model_dir /path/to/TDYSN/models \
    --data_root /path/to/dataset
```

**Add this footnote to the ablation table in the paper:**
```latex
\footnote{AUC-Borji values corrected from earlier computation;
a sign error in \texttt{np.trapz} integration direction
(\texttt{auc = np.trapz(tp, fp)} $\to$ \texttt{-np.trapz(tp, fp)})
has been fixed.}
```

---

## Task 4 — Final Test Set Evaluation

**What:** Re-run evaluation on the test split using the corrected metrics to confirm the numbers in Table 3.  
**Context:** The notebook achieved on the 50-epoch checkpoint: CC=0.6423, KL=0.9270, SIM=0.5010, NSS=3.4583, AUC-Borji=0.9486 (corrected sign).

**Create:** `scripts/evaluate_final.py`

```python
"""
Evaluate the best TDSal checkpoint on the held-out test set.
Uses the same 70/15/15 split with seed=42 as the notebook.

Usage:
    python scripts/evaluate_final.py \
        --checkpoint checkpoints/tdsp_50epoch.pth \
        --data_root /path/to/dataset
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

# Import corrected metric functions from recompute_ablation or a shared metrics.py
# from scripts.recompute_ablation import cc_metric, kl_metric, sim_metric, nss_metric, auc_borji_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    from model import YOLOTaskSaliencyModel
    from dataset import TaskSaliencyDataset

    task_mapping = {
        "task1": "free view", "task2": "count people",
        "task3": "detect the emotion", "task4": "identify the action",
    }
    transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])
    dataset = TaskSaliencyDataset(args.data_root, task_mapping, transform, transform)

    total = len(dataset)                   # 1968
    train_size = int(0.70 * total)         # 1377
    val_size = int(0.15 * total)           # 295
    test_size = total - train_size - val_size  # 296

    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # num_workers=2 to avoid the warning from the notebook (recommended max=2 on Colab)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOTaskSaliencyModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    print(f"Test set size: {len(test_ds)}")
    print(f"Checkpoint:    {args.checkpoint}")
    print("Running evaluation (this takes a few minutes for AUC-Borji)...")

    # Use evaluate_saliency_model from eval.py — ensure AUC sign is fixed there
    from eval import evaluate_saliency_model
    metrics = evaluate_saliency_model(model, test_loader, device)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("\n-> Update Table 3 in paper0005.tex with these values if they differ.")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python scripts/evaluate_final.py \
    --checkpoint checkpoints/tdsp_50epoch.pth \
    --data_root /path/to/dataset
```

---

## Known Bug to Fix Before Running Anything

The notebook's full training loop with `evaluate_saliency_model` during training crashed with:

```
RuntimeError: Inference tensors cannot be saved for backward.
```

This is because `SentenceTransformer.encode()` runs in `torch.inference_mode()` by default. The fix is already shown in the notebook's ablation `TaskEncoderSBERT`:

```python
# In TaskEncoderSBERT.forward() (and equivalently in your TaskEncoder.forward()):
embeddings = self.text_encoder.encode(task_descriptions, convert_to_tensor=True)
embeddings = embeddings.detach().clone()   # <- this line fixes the bug
return F.relu(self.linear(embeddings))
```

**Make sure this fix is present in `model.py` before running any of the scripts above.**

---

## Summary

| Script | Output | Where it goes in the paper |
|---|---|---|
| `compute_efficiency.py` | Param count (M), FLOPs (GFLOPs), inference time (ms) | Table 1 + Section 3.2 paragraph |
| `verify_backbone_shapes.py` | Backbone output shape confirmation | Section 3.2.1 clarification |
| `recompute_ablation.py` | Corrected ablation table (LaTeX ready) | Ablation table (replace existing) |
| `evaluate_final.py` | Final test metrics (5 metrics) | Table 3 verification |

**Paper deadline: May 31, 2026.**
