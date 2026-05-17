# Changes: `TDSal.ipynb` → `TDSal_CameraReady.ipynb`

This document covers every architectural difference and bug fix made when producing the
camera-ready notebook from the original training notebook.

---

## 1. Environment: Colab → Local VSCode

| Aspect | `TDSal.ipynb` | `TDSal_CameraReady.ipynb` |
|--------|--------------|--------------------------|
| Runtime | Google Colab | Local VSCode Jupyter kernel |
| Storage | Google Drive (`/content/drive/…`) | Local filesystem |
| Drive mount | `from google.colab import drive; drive.mount(…)` | Removed |
| Pip style | `!pip install …` | `%pip install …` (kernel-aware) |
| Path config | Hard-coded Colab paths scattered across cells | Single **Path Configuration** cell at the top |
| `sentence-transformers` | Unpinned (pulls v4.x → `torchcodec` crash) | Pinned to `<4.0.0` (avoids the FFmpeg/torchcodec dependency) |

**Why `sentence-transformers<4.0.0`?**  
Version 4.0 introduced multimodal support that depends on `torchcodec`, which in turn
requires FFmpeg shared libraries. On macOS (and many Linux environments without FFmpeg)
this raises a `RuntimeError` at import time — before any model code even runs. Pinning
to `<4.0.0` avoids the dependency entirely with no functional impact on text embeddings.

---

## 2. Notebook Structure

| `TDSal.ipynb` (95 cells) | `TDSal_CameraReady.ipynb` (28 cells) |
|--------------------------|--------------------------------------|
| Training loops (3 variants, cells 44 / 49 / 52) | **Removed** — inference-only |
| Checkpoint save logic | **Removed** |
| Ablation training (`train_and_save`, cells 64–67) | **Removed** — evaluation-only |
| Scattered imports across many cells | Consolidated into a single imports cell |
| `YOLOBackbone` defined twice (cells 20 and 56) | One clean `YOLOBackbone` + private `_YOLOBackbone` for ablation |
| `device = …` set in multiple cells | Set once in the imports cell |
| `num_workers=4` | Changed to `num_workers=2` (safe default; 4 causes warnings) |
| `sklearn.metrics.roc_auc_score` dependency | Removed; pure-numpy AUC implementation used throughout |

---

## 3. Bug Fixes

### 3.1 `TaskEncoder.forward()` — inference-mode tensor crash

**File:** `TaskEncoder` class (cell 26 in original, cell 13 in camera-ready)

**Original:**
```python
def forward(self, task_descriptions):
    embeddings = self.text_encoder.encode(task_descriptions, convert_to_tensor=True)
    embeddings = self.linear(embeddings)
    return F.relu(embeddings)
```

**Fixed:**
```python
def forward(self, task_descriptions):
    embeddings = self.text_encoder.encode(task_descriptions, convert_to_tensor=True)
    embeddings = embeddings.detach().clone()   # exits inference_mode graph
    return F.relu(self.linear(embeddings))
```

**Root cause:** `SentenceTransformer.encode()` runs internally under
`torch.inference_mode()`. The returned tensor carries the inference-mode flag, so
PyTorch refuses to save it for the autograd backward pass:
```
RuntimeError: Inference tensors cannot be saved for backward.
```
`.detach().clone()` converts it to an ordinary leaf tensor. This fix already existed
in the ablation's `TaskEncoderSBERT` but was missing from the main `TaskEncoder`.

---

### 3.2 AUC-Borji sign error

**File:** `auc_borji_metric()` function

**Original:**
```python
auc = np.trapz(tp, fp)   # returns a negative number
```

**Fixed:**
```python
auc = -np.trapz(tp, fp)  # correct positive area in [0, 1]
```

**Root cause:** As the threshold increases from 0 → 1, both `tp` (true-positive rate)
and `fp` (false-positive rate) decrease monotonically. `np.trapz(y, x)` integrates
left-to-right; when `x` is a decreasing sequence the integral comes out negative.
Negating gives the correct area under the ROC curve, which must lie in `[0, 1]`.

The fix was applied in **three places**:
1. `auc_borji_metric()` called by `evaluate_saliency_model()` (Section C)
2. The ablation evaluation loop (Section D)
3. The `_TaskEncoderSBERT` in the ablation variant also received the `detach().clone()` fix

---

## 4. New Sections (Reviewer Responses)

### 4.1 Section A — Efficiency Metrics *(Reviewer 1)*

Added functions that were entirely absent from the original:

- `count_parameters(model)` — counts trainable parameters only
- `measure_inference_time(model, …, n_runs=100, warmup=10)` — CUDA-synchronised timing
- FLOPs via `thop.profile()` using a `_Wrap` shim that fixes the task string as a
  constant, making the forward pass purely tensor-based for tracing

Output: parameter count, FLOPs, CPU inference time, GPU inference time — ready for
Table 1 and Section 3.2 of the paper.

### 4.2 Section B — Backbone Shape Verification *(Reviewer 1)*

Added runtime shape assertions that confirm the backbone is **single-scale only**:

```python
assert backbone_out.shape == (1, 512, 12, 12)  # single scale, stride-32
assert fpn_out.shape      == (1, 128, 12, 12)  # 512→128 channel reduction
```

Also added a markdown table of all 10 backbone layers (0–9) with types, channel
dimensions, and cumulative strides. This directly answers the reviewer question:
*"it is unclear how many multiscale YOLO features are fused before the FPM."*
(Answer: zero — only a single-scale SPPF output is passed to the FPM.)

### 4.3 Section C — Unified Evaluation with LaTeX Output *(All reviewers)*

The original had two separate, inconsistent evaluation paths:
- Cell 68: `auc_borji_metric()` with the sign bug
- Cell 91: `evaluate_saliency_model()` using `sklearn.metrics.roc_auc_score` (different
  implementation)

The camera-ready notebook consolidates into one `evaluate_saliency_model()` that calls
the corrected `cc_metric`, `kl_metric`, `sim_metric`, `nss_metric`, and
`auc_borji_metric` — ensuring the same metric code is used for validation, ablation,
and final test evaluation. A formatted LaTeX row is printed at the end for direct
copy-paste into Table 3.

### 4.4 Section D — Ablation Re-Evaluation with LaTeX Table *(Reviewers 1 & 3)*

Replaced the `train_and_save()` training loop with an evaluation-only pass over
existing ablation checkpoints. Added `pandas` DataFrame construction and `.to_latex()`
output. Skips gracefully if checkpoint files are absent.

---

## 5. Minor Fixes

| Item | Original | Camera-Ready |
|------|----------|--------------|
| YOLO model name | `yolov5s.pt` in some cells, `yolov5su.pt` in others | Unified to `yolov5su.pt` throughout |
| `DataLoader(num_workers=…)` | 4 | 2 |
| Augmentation at inference | Applied in some evaluation paths | Explicitly disabled (`paired_transforms=None`) |
| `torch.load()` warning | No `weights_only` arg | `map_location=device` kept; upgrade path left to user |
| `CKPT_PATH` | `tdsp_50epoch.pth` (Google Drive) | `tdsp_40epoch.pth` (local models folder) |

---

## 6. Required Paper Changes (`Can_Mizrakli_TDSal.pdf`)

The PDF cannot be edited directly (no LaTeX source in the repository). The issues below
must be corrected in the paper source before resubmission. They are listed in priority order.

### 6.1 CRITICAL — Table 4: AUC-Borji sign error propagated into the paper

**Location:** Table 4 (Ablation Study, page 10)

The AUC-Borji values in the published table are **negative**, which is physically
impossible (AUC is bounded [0, 1]). This is the same sign bug from the code
(`np.trapz` without negation) that was fixed in the camera-ready notebook.

| Model | AUC-B in paper | Corrected value |
|-------|---------------|-----------------|
| Full TDSal | −0.9264 | **+0.9264** |
| w/o Task | −0.9005 | **+0.9005** |
| w/o Transformer | −0.9197 | **+0.9197** |
| w/o SBERT | −0.9079 | **+0.9079** |
| w/o FPM | −0.9098 | **+0.9098** |

Also fix the column header: `AUC-B ↓` (lower is better) → `AUC-B ↑` (higher is better).

Run Section D of the camera-ready notebook to obtain the corrected values from the
actual checkpoints.

### 6.2 CRITICAL — "multiscale features" is factually wrong throughout the paper

The backbone is truncated at layer 9 (SPPF) and produces a **single feature map**
at stride-32 (`[B, 512, 12, 12]`). There is no multi-scale pyramid before the FPM.

Every occurrence of "multiscale" that refers to the YOLO features must be corrected:

| Location | Current text | Corrected text |
|----------|-------------|----------------|
| Abstract | "YOLO-derived multiscale visual features" | "YOLO-derived single-scale visual features" |
| Section 1, Contributions (1st bullet) | "YOLO-derived multiscale visual features with Sentence-BERT…" | "YOLO-derived single-scale visual features (stride-32) with Sentence-BERT…" |
| Section 6.2, Table 3 discussion | "YOLO-derived multiscale features through a transformer" | "YOLO-derived single-scale features through a transformer" |
| Section 6.3, Ablation discussion (last paragraph) | "multi-scale feature fusion components act as regularizing mechanisms" | "feature projection and fusion components act as regularizing mechanisms" |

Section B of the camera-ready notebook produces the runtime proof with shape assertions
and a layer-by-layer table to include as a footnote or appendix.

### 6.3 IMPORTANT — Checkpoint epoch count discrepancy

**Location:** Section 3.4 ("We train our TDSal model for 50 epochs")

The only checkpoint available locally is `tdsp_40epoch.pth`. If the evaluation in
Section C is run with this file, the metrics in Table 2 will differ from the
paper's reported values (which were computed on the 50-epoch model).

**Options:**
- If the 50-epoch checkpoint still exists on Google Drive, download it and rename it
  to `tdsp_50epoch.pth` — then update `CKPT_PATH` in the notebook accordingly.
  The paper text requires no change.
- If only the 40-epoch checkpoint survives, update Section 3.4 to say 40 epochs,
  re-run Section C, and update Table 2 with the new metrics.

### 6.4 IMPORTANT — Efficiency metrics missing (Reviewer 1)

**Location:** Table 1 (Implementation Details, page 5) and Section 3.2

Table 1 currently lists only architectural hyperparameters. Run Section A of the
camera-ready notebook and add a new row (or a companion table) with:

| Metric | Value (to be filled after running Section A) |
|--------|----------------------------------------------|
| Trainable parameters | — M |
| FLOPs (single 384×384 image) | — G |
| Inference time (CPU, n=100) | — ± — ms |
| Inference time (GPU, n=100) | — ± — ms |

Add a paragraph to Section 3.2 citing these numbers.
