# Code Changes Log — TDSal Camera-Ready (CGI 2026)

## [2026-05-11] Rebuilt `TDSal_CameraReady.ipynb` — inference-only Colab version

Replaced the previous draft (which still included training cells) with a clean
inference-only notebook that runs top-to-bottom on Colab without retraining.
Key structural changes versus the previous draft:
- Removed all training loops and checkpoint-save cells
- Ablation section changed to evaluation-only (skips gracefully if `.pth` files absent)
- All file paths corrected to match the original `TDSal.ipynb`:
  - Dataset: `/content/drive/MyDrive/TDYSN/Task-based-eye-fixation-dataset_1024x768`
  - Checkpoint: `/content/drive/MyDrive/TDSP/models/tdsp_50epoch.pth`
  - Ablation models: `/content/drive/MyDrive/TDYSN/models/`
- `TaskSaliencyDataset.__getitem__` now stores the task key (e.g. `"task1"`) in
  `samples`, resolving to the description string at `__getitem__` time — matching
  the original exactly so the 70/15/15 seed-42 split reproduces the same test set
- `DataLoader(num_workers=2)` (was 4 in original; 2 is the safe Colab max)
- Paired augmentations disabled at inference (`paired_transforms=None`)
- LaTeX output added to Sections A and B (snippet ready to paste into paper)

Changes are recorded chronologically. Each entry states what changed, where, and why.

---

## [2026-05-11] Created `TDSal_CameraReady.ipynb`

New notebook derived from `TDSal.ipynb`. All changes below are relative to the original.

---

### Change 1 — Fix: `TaskEncoder.forward()` — inference-mode tensor crash

**Location:** Cell 16 (`TaskEncoder` class)  
**Reviewer that triggered this:** All (prerequisite for running any training)

**Original code (`TDSal.ipynb`, Cell 26):**
```python
def forward(self, task_descriptions):
    embeddings = self.text_encoder.encode(task_descriptions, convert_to_tensor=True)
    embeddings = self.linear(embeddings)
    return F.relu(embeddings)
```

**Fixed code (`TDSal_CameraReady.ipynb`):**
```python
def forward(self, task_descriptions):
    embeddings = self.text_encoder.encode(task_descriptions, convert_to_tensor=True)
    embeddings = embeddings.detach().clone()   # CRITICAL: exits inference_mode graph
    return F.relu(self.linear(embeddings))
```

**Why:** `SentenceTransformer.encode()` executes under `torch.inference_mode()` internally. The resulting tensor cannot be saved for the autograd backward pass, causing:
```
RuntimeError: Inference tensors cannot be saved for backward.
```
Adding `.detach().clone()` converts the tensor into an ordinary leaf tensor that autograd can handle. This fix was already present in the ablation's `TaskEncoderSBERT` (Cell 58 of original) but was missing from the main `TaskEncoder`.

---

### Change 2 — New Section: Efficiency Metrics (Camera-Ready Task 1)

**Location:** Cell 25 in `TDSal_CameraReady.ipynb` (new section "6. Efficiency Metrics")  
**Reviewer that triggered this:** Reviewer 1 — *"the paper claims to be lightweight but does not provide complexity measurements such as number of parameters, inference speed, or FLOPs"*

**New code added:**
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, images, task_descs, device, n_runs=100, warmup=10):
    # ... warmup + timed loop with CUDA synchronize
    return np.mean(times), np.std(times)

def compute_efficiency_metrics(checkpoint_path=None):
    # Loads model, counts params, runs thop.profile() for FLOPs,
    # measures CPU and GPU inference time (100 runs, 10 warmup)
```

The FLOPs computation wraps the model in a `_Wrap` class that fixes the task description to `"free view"` as a constant string, making the forward pass purely tensor-based so `thop.profile()` can trace it.

**Paper target:** Numbers go into **Table 1** and a new efficiency paragraph in **Section 3.2**.

---

### Change 3 — New Section: Backbone Shape Verification (Camera-Ready Task 2)

**Location:** Cell 27 in `TDSal_CameraReady.ipynb` (new section "7. Backbone Shape Verification")  
**Reviewer that triggered this:** Reviewer 1 — *"it is unclear how many multiscale YOLO features are fused before the FPM"*

**New code added:**
```python
def verify_backbone_shapes():
    model = YOLOTaskSaliencyModel()
    model.eval()
    dummy = torch.randn(1, 3, 384, 384)
    backbone_out = model.backbone(dummy)   # [1, 512, 12, 12]
    fpn_out      = model.fpn(backbone_out) # [1, 128, 12, 12]
    assert backbone_out.shape == (1, 512, 12, 12)
    assert fpn_out.shape      == (1, 128, 12, 12)
```

**Accompanying markdown table** documents all 10 backbone layers (0–9) with their type, channel dimensions, and strides. It explicitly clarifies that **only a single-scale output** from SPPF (layer 9) is passed to the FPM — there is no multi-scale pyramid before the FPM.

**Paper target:** Replaces vague "multiscale" phrasing in **Section 3.2.1**.

---

### Change 4 — Fix: AUC-Borji sign error (Camera-Ready Task 3)

**Location:** `auc_borji_metric()` function in Cell 31 of `TDSal_CameraReady.ipynb`  
**Reviewer that triggered this:** Reviewer 1 and Reviewer 3 (comprehensive evaluation concern)

**Original code (`TDSal.ipynb`, Cell 68):**
```python
auc = np.trapz(tp, fp)   # WRONG: returns negative because fp is decreasing
```

**Fixed code:**
```python
auc = -np.trapz(tp, fp)  # CORRECT: negating gives positive area in [0, 1]
```

**Why:** In the ROC curve loop, `fp` (false-positive rate) decreases monotonically as the threshold increases from 0 to 1. `np.trapz(y, x)` computes the integral from left to right; when `x` is decreasing, the result is negative. Negating gives the correct positive area. AUC-Borji is bounded `[0, 1]` — any negative value is a sign error.

The same fix is propagated into:
- `auc_borji_metric()` used by `evaluate_saliency_model()` (Cell 32)
- `evaluate_ablation_model()` inside the ablation evaluation (Cell 45)
- The `_TaskEncoderSBERT` in the ablation variant also carries the `detach().clone()` fix

**Paper target:** A footnote is added to the ablation table noting this correction.

---

### Change 5 — New: Corrected `evaluate_saliency_model()` with unified metrics

**Location:** Cell 32 in `TDSal_CameraReady.ipynb`  
**Reviewer that triggered this:** All reviewers (evaluation quality)

The original notebook had two separate evaluation functions:
- Cell 91: `evaluate_saliency_model()` — used `auc_borji()` from sklearn ROC (different implementation)
- Cell 68: `auc_borji_metric()` — the batch-form implementation with the sign bug

The camera-ready notebook consolidates into a **single** `evaluate_saliency_model()` that calls the corrected `cc_metric`, `kl_metric`, `sim_metric`, `nss_metric`, and `auc_borji_metric` (all defined in the same cell block, all batch-vectorized). This ensures consistency between training-time validation metrics, ablation evaluation, and final test evaluation.

---

### Change 6 — New: Ablation evaluation with LaTeX output (Camera-Ready Task 3, continued)

**Location:** Cell 45 in `TDSal_CameraReady.ipynb`  
**Reviewer that triggered this:** Reviewer 1 and Reviewer 3

Added `pandas` DataFrame construction and `.to_latex()` call at the end of the ablation evaluation block:

```python
df = pd.DataFrame(rows)[['Model','CC','KL','SIM','NSS','AUC-Borji']].round(4)
latex = df.to_latex(
    index=False,
    caption=r'Ablation study...\footnote{AUC-Borji corrected: np.trapz -> -np.trapz}',
    label='tab:ablation',
    column_format='lccccc'
)
print(latex)
```

This outputs a LaTeX-ready ablation table with the corrected values and an inline footnote explaining the sign fix.

---

### Change 7 — New: Final test-set evaluation with LaTeX row output (Camera-Ready Task 4)

**Location:** Cells 47–48 in `TDSal_CameraReady.ipynb` (section "13. Final Test-Set Evaluation")  
**Reviewer that triggered this:** All reviewers

Added a dedicated final evaluation section that:
1. Loads `tdsp_50epoch.pth` (the best checkpoint)
2. Runs `evaluate_saliency_model()` (with corrected AUC-Borji) on the held-out test split (seed=42)
3. Prints results in a formatted table
4. Outputs a LaTeX table row for direct copy-paste into Table 3 of the paper

```python
print(f"TDSal (Ours) & {m['CC']:.4f} & {m['KL']:.4f} & "
      f"{m['SIM']:.4f} & {m['NSS']:.4f} & {m['AUC-Borji']:.4f} \\\\")
```

Expected output (reference from notebook): CC=0.6423, KL=0.9270, SIM=0.5010, NSS=3.4583, AUC-Borji=0.9486

---

### Structural / Cleanup Changes

| Change | Original | Camera-Ready |
|--------|----------|--------------|
| Duplicate training loop cells (Cells 44, 49, 52) | Three separate training cells with overlapping code | Single unified training loop with val metric tracking |
| Duplicate YOLOBackbone definitions (Cells 20, 56) | Two separate class definitions | One clean `YOLOBackbone` + private `_YOLOBackbone` for ablation |
| Google Colab drive mount | Cell 4 | Cell 3 (first code cell) |
| `num_workers=4` in DataLoader | Original (causes warnings on Colab) | Changed to `num_workers=2` |
| YOLO model name | `yolov5s.pt` (some cells) vs `yolov5su.pt` (others) | Unified to `yolov5su.pt` throughout |
| `sklearn.metrics.roc_auc_score` dependency | Used in `auc_borji()` (Cell 90) | Removed; pure numpy implementation used |

---

## Summary Table

| # | What | Where in new notebook | Reviewer |
|---|------|-----------------------|----------|
| 1 | Bug fix: `detach().clone()` in `TaskEncoder` | Cell 16 | — |
| 2 | New: efficiency metrics (params, FLOPs, time) | Cells 24–25 | R1 |
| 3 | New: backbone shape verification | Cells 26–27 | R1 |
| 4 | Bug fix: AUC-Borji sign (`-np.trapz`) | Cell 31 | R1, R3 |
| 5 | New: unified corrected `evaluate_saliency_model` | Cell 32 | All |
| 6 | New: ablation LaTeX table with footnote | Cell 45 | R1, R3 |
| 7 | New: final evaluation + LaTeX row | Cells 47–48 | All |
