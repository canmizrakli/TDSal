# Camera-Ready Reviewer Fixes: Change Log for Code Submission

## New notebook artifacts

Two new notebooks were created by combining the original workflow context from `TDSal.ipynb` with the camera-ready reviewer-fix structure from `TDSal_CameraReady_colab.ipynb`:

1. `TDSal_ReviewerFixes_local.ipynb` (local-machine paths and install flow)
2. `TDSal_ReviewerFixes_colab.ipynb` (Google Colab + Drive-aware setup)

Both notebooks are synchronized to the same evaluation/model logic so results are comparable across environments.

## What was changed and how each reviewer point was addressed

### 1) SBERT inference-mode bug fix in `TaskEncoder.forward()`

Reviewer point: fix the crash caused by passing inference-mode tensors into a trainable linear layer.

Applied change in both new notebooks:

- In `TaskEncoder.forward`, task embeddings now use:
  - `emb = emb.detach().clone().to(dev)`
- This ensures:
  - the tensor is detached from SBERT's inference-mode context,
  - a fresh clone is used for safe downstream autograd behavior,
  - tensor/device consistency is preserved before the projection layer.

Also aligned in ablation SBERT encoder (`_TaskEncoderSBERT.forward`) to keep behavior consistent in Section D.

### 2) `compute_efficiency.py` requirement (params, FLOPs, inference time)

Reviewer point: include model efficiency analysis.

Applied change in both new notebooks (Section A):

- Added explicit script-equivalent labeling: `compute_efficiency.py equivalent`.
- Kept/standardized the efficiency cell to report:
  - trainable parameter count,
  - FLOPs (single-image) via `thop.profile`,
  - CPU inference latency (mean ± std),
  - optional GPU inference latency (mean ± std when CUDA exists).
- Colab flow retains robust FLOPs wrapping and CPU-safe profiling behavior.

### 3) `verify_backbone_shapes.py` requirement (YOLO output shape confirmation)

Reviewer point: verify backbone output shape for 384x384 input.

Applied change in both new notebooks (Section B):

- Added explicit script-equivalent labeling: `verify_backbone_shapes.py equivalent`.
- Verification cell prints:
  - input shape,
  - backbone output shape,
  - FPM output shape,
  - backbone layer count.
- Assertions enforce expected values:
  - backbone: `(1, 512, 12, 12)`
  - FPM: `(1, 128, 12, 12)`
- A ready-to-paste LaTeX clarification snippet for Section 3.2.1 remains included.

### 4) AUC-Borji sign error fix + ablation rerun

Reviewer point: correct sign bug in AUC-Borji integration and recompute ablations.

Applied change in both new notebooks (Sections C + D):

- `auc_borji_metric` uses corrected integration direction:
  - `aucs.append(-_trapz(tp, fp))`
- Compatibility guard retained:
  - `_trapz = getattr(np, 'trapezoid', np.trapz)`
  - supports NumPy versions where `np.trapz` is deprecated/removed.
- Ablation section is explicitly labeled as corrected sign rerun and uses the same fixed metric path.

### 5) `evaluate_final.py` requirement (final test metrics check)

Reviewer point: verify test-set metrics against paper values.

Applied change in both new notebooks (Section C):

- Added explicit script-equivalent labeling: `evaluate_final.py equivalent`.
- Final evaluation cell:
  - runs full test evaluation,
  - prints all five metrics (CC, KL, SIM, NSS, AUC-Borji),
  - compares against reference values:
    - `CC=0.6423, KL=0.9270, SIM=0.5010, NSS=3.4583, AUC-Borji=0.9486`.
- Kept LaTeX table-row output for direct manuscript insertion.

## Cross-notebook consistency changes

To avoid local/Colab divergence, both notebooks were normalized to the same core logic and reviewer mapping:

- Same model/evaluation/ablation code path in both variants.
- Same section labels and checklist framing.
- Same metric computation and corrected AUC implementation.

## Environment-specific setup differences

### `TDSal_ReviewerFixes_local.ipynb`

- Uses explicit local absolute paths:
  - dataset root,
  - main checkpoint,
  - ablation checkpoint directory.
- Includes local package install cell suitable for desktop/Jupyter usage.

### `TDSal_ReviewerFixes_colab.ipynb`

- Uses Colab Drive mount + candidate-path resolution.
- Preserves Colab-friendly install behavior and warnings for missing paths/checkpoints.

## Additional technical cleanup performed

- Corrected LaTeX row printing to emit a proper table newline (`\\`) in both new notebooks.
- Section titles now explicitly name script equivalence to match reviewer checklist terminology.

## Deliverable summary

Created files for submission:

1. `TDSal_ReviewerFixes_local.ipynb`
2. `TDSal_ReviewerFixes_colab.ipynb`
3. `changes_for_review.md` (this file)
