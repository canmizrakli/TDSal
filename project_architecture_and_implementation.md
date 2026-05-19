# TDSal Project: Full Architecture and Implementation Reference

## 1. Project Scope

TDSal is a task-driven visual saliency prediction system that conditions saliency estimation on natural-language task descriptions (e.g., `free view`, `count people`, `detect the emotion`, `identify the action`).

This project now includes complete end-to-end pipeline notebooks for both environments:

1. `TDSal_CompletePipeline_local.ipynb`
2. `TDSal_CompletePipeline_colab.ipynb`

Each notebook covers training, evaluation, efficiency analysis, backbone-shape verification, corrected AUC-Borji computation, and ablation re-evaluation.

## 2. Dataset and Supervision

### 2.1 Dataset layout

The pipeline assumes the task-based eye-fixation dataset with this layout:

- `stimuli/` contains source RGB images (`.jpg`/`.png`)
- `task1/fdm`, `task2/fdm`, `task3/fdm`, `task4/fdm` contain saliency/fixation density maps (`.png`)

### 2.2 Task semantic mapping

Folder-to-text mapping is explicit and deterministic:

- `task1 -> free view`
- `task2 -> count people`
- `task3 -> detect the emotion`
- `task4 -> identify the action`

These task descriptions are passed to the text encoder to condition saliency prediction.

### 2.3 Split protocol

Data is split 70/15/15 into train/val/test using:

- `torch.Generator().manual_seed(42)`

This split is reproducible and consistent with the reviewer notebooks.

### 2.4 Augmentation policy

Two dataset instances are used:

- `dataset_aug` for train only (paired transforms)
- `dataset_plain` for val/test (no stochastic augmentation)

Paired transforms ensure the same random transformation is applied to both input image and saliency map:

- random horizontal flip
- random rotation

This prevents image/saliency misalignment during supervised training.

## 3. Model Architecture

## 3.1 High-level graph

`Image + task text -> backbone -> FPM -> task encoder -> transformer fusion -> decoder -> saliency map`

## 3.2 YOLO backbone (visual features)

- Backbone source: `ultralytics` YOLOv5su model
- Truncation: layers `0..9` (up to SPPF), detection head removed
- Output feature map for `384x384` input: `[B, 512, 12, 12]`

The notebooks include explicit assertion checks for this shape.

## 3.3 Feature Projection Module (FPM)

- A `1x1` convolution projects visual channels:
- `512 -> 128`
- Resulting map shape: `[B, 128, 12, 12]`

Purpose: reduce channel dimensionality and stabilize downstream fusion/decoder cost.

## 3.4 Task encoder (SBERT)

- Text model: `all-MiniLM-L6-v2` (384-d embeddings)
- Projection: linear layer `384 -> task_embed_dim` (default 64)
- Nonlinearity: `ReLU`

### Critical reviewer bug fix

SBERT `encode()` can produce inference-mode tensors. Before the trainable linear layer, embeddings are now transformed with:

- `emb = emb.detach().clone().to(dev)`

This prevents runtime autograd failures and fixes the inference-mode crash path.

## 3.5 Vision-language fusion

- Task embedding is projected to query dimension
- Visual map is flattened to token sequence `(H*W, B, C)`
- Query token is concatenated with visual tokens
- Transformer encoder (single layer by default) processes tokens
- Output tokens are reshaped back to `[B, C, H, W]`

This allows task-conditioned global interaction over spatial features.

## 3.6 Saliency decoder

Decoder structure:

1. `Conv2d(in=128,out=64,k=3,p=1)` + ReLU
2. `ConvTranspose2d(64->32,k=4,s=2,p=1)` + ReLU
3. `ConvTranspose2d(32->1,k=4,s=2,p=1)` + Sigmoid

Output is a normalized saliency map in `[0,1]`.

## 4. Loss and Optimization

## 4.1 Composite saliency loss

`SaliencyLoss = alpha * KL(gt || pred) + beta * (1 - CC(pred, gt))`

- `KL` aligns predicted and target saliency distributions
- `CC` enforces structural/statistical agreement

Default weights:

- `alpha = 1.0`
- `beta = 1.0`

## 4.2 Training loop behavior

For each batch:

1. forward pass with image batch + task descriptions
2. up/downsample predictions to target map resolution
3. compute composite loss
4. backward pass + optimizer step

Validation loss is computed each epoch with `model.eval()` and `torch.no_grad()`.

## 4.3 Optimizer and defaults

- Optimizer: Adam
- Base LR: `1e-4`
- Default training epochs: `50`

Control flags in complete pipeline notebooks:

- `RUN_TRAINING` (bool)
- `TRAIN_EPOCHS`
- `TRAIN_LR`
- `TRAIN_OUTPUT_CKPT`

## 5. Evaluation Metrics

The evaluation module reports:

1. CC (Pearson correlation)
2. KL divergence
3. SIM (histogram intersection)
4. NSS
5. AUC-Borji

### 5.1 AUC-Borji correction

AUC-Borji integration uses corrected sign handling because ROC traversal direction causes negative area if integrated directly:

- Corrected: `auc = -trapz(tp, fp)`

Implementation uses compatibility fallback for NumPy versions:

- `_trapz = getattr(np, 'trapezoid', np.trapz)`

## 6. Reviewer-Oriented Analysis Blocks

The complete pipeline notebooks include explicit equivalents for reviewer-requested scripts:

1. `compute_efficiency.py` equivalent
- trainable parameter count
- FLOPs (THOP profile)
- inference time (CPU/GPU)

2. `verify_backbone_shapes.py` equivalent
- explicit backbone/FPM shape print and assertions

3. `evaluate_final.py` equivalent
- final test metrics with paper-reference comparison:
  - `CC=0.6423`
  - `KL=0.9270`
  - `SIM=0.5010`
  - `NSS=3.4583`
  - `AUC-Borji=0.9486`

4. ablation re-evaluation block
- evaluates `full`, `w/o Task`, `w/o Transformer`, `w/o SBERT`, `w/o FPM`
- all with corrected AUC-Borji sign

## 7. Ablation Architecture Variants

Ablation model (`TDYSN`) toggles four factors:

- task conditioning on/off
- transformer fusion vs FiLM-like conditioning
- SBERT encoder vs learned task-id embedding
- FPM on/off

This isolates architectural contribution of each subsystem.

## 8. Environment Design (Local vs Colab)

## 8.1 Local notebook

`TDSal_CompletePipeline_local.ipynb`:

- uses explicit local absolute paths
- warns if evaluation checkpoint is missing
- supports training-from-scratch to generate checkpoint

## 8.2 Colab notebook

`TDSal_CompletePipeline_colab.ipynb`:

- mounts Google Drive
- resolves candidate dataset/checkpoint/model directories
- prints available checkpoint files when not found
- includes colab-compatible install setup

## 9. Reproducibility and Stability Choices

1. Fixed split seed (`42`)
2. Explicit train/val/test loader separation
3. `safe_torch_load()` fallback behavior for torch version differences
4. Fixed shape assertions for backbone and FPM outputs
5. Deterministic metric definitions implemented in notebook cells

## 10. Main Deliverables in Repository

1. `TDSal_CompletePipeline_local.ipynb`
2. `TDSal_CompletePipeline_colab.ipynb`
3. `TDSal_ReviewerFixes_local.ipynb`
4. `TDSal_ReviewerFixes_colab.ipynb`
5. `changes_for_review.md`
6. `project_architecture_and_implementation.md`

## 11. Practical Execution Flow

Recommended order in complete pipeline notebooks:

1. setup paths and install deps
2. run imports and dataset/split cells
3. run model definitions
4. choose `RUN_TRAINING` mode
5. run efficiency and shape verification
6. run corrected final evaluation
7. run ablation section (if ablation checkpoints exist)
8. run qualitative visualization

This sequence provides both model-development completeness and reviewer-facing evidence in one reproducible pipeline.
