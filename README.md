# TDSal – YOLO-Based Top-Down Saliency Network

**TDSal** is a lightweight deep-learning model that predicts *task-driven visual saliency maps*.  
Developed as an academic research project **under the supervision of Prof. Dr. Tolga Kurtuluş Çapın**, TDSal fuses multiscale visual features from a pre-trained YOLO backbone with semantic task embeddings from Sentence-BERT via a transformer fusion block, and decodes the result into high-resolution saliency maps.

---

## Project Highlights

- **Top-down attention** – accepts explicit task prompts (e.g., “count people”, “detect action”) to steer gaze prediction.  
- **Modular pipeline**  
  - **YOLOv5-su backbone** → rich multiscale visual features  
  - **Feature Projection Module (FPM)** → 1 × 1 convolutions to compress channels  
  - **Sentence-BERT task encoder** → 64-D semantic vector  
  - **Lightweight transformer fusion** → 1 encoder layer, 4 heads  
  - **Saliency decoder** → up-convolutions + sigmoid saliency map  
- **Strong performance** on a 1 968-image eye-tracking dataset  
  - **NSS = 3.53**, **AUC-Borji = 0.9489**, **CC = 0.643**, **KL = 0.924**, **SIM = 0.511**  
- **End-to-end training** in < 40 epochs on a single GPU (~40 k iterations).  
- **Compact model** – < 15 M parameters, deployable in real-time systems.

---

## Dataset

Model training and evaluation use the **Task-Driven Eye-Fixation Dataset** (Albayrak 2020):

| Split | Images | Tasks (“free”, “count people”, “detect emotion”, “detect action”) |
|-------|--------|-------------------------------------------------------------------|
| Train | 1 377  | 4 |
| Val   |   295  | 4 |
| Test  |   296  | 4 |

Fixation maps are provided at 48 × 48 resolution.

---

## Results

| Metric        | TDSal (Ours) |
|---------------|--------------|
| **NSS**       | **3.53** |
| **AUC-Borji** | **0.9489** |
| CC            | 0.6433 |
| KLDiv ↓       | 0.9239 |
| SIM           | 0.5112 |
