# 🧠 Few-Shot Generative Modelling Architectures Analysis

[![Weights](https://img.shields.io/badge/Weights-Hugging_Face-FFD21E.svg?style=flat-square&logo=huggingface)](https://huggingface.co/VishyRK/Few-Shot-GM-DC-Weights/tree/main)
[![Report](https://img.shields.io/badge/Report-63_Page_PDF-red.svg?style=flat-square)](#)

## 📖 1. Project Abstract / Overview

Generative Modelling under Data Constraint (GM-DC) is one of the most significant challenges in modern deep learning. When a Generative Adversarial Network (GAN) is given an extremely limited dataset (10–100 samples), the discriminator rapidly overfits, preventing the generator from learning a meaningful data manifold.

This project presents a deep, comparative analysis of state-of-the-art architectures designed specifically for extreme low-shot regimes. We evaluated four core architectures:

- **StyleGAN2-ADA** (Adaptive Discriminator Augmentation)
- **DANI** (Data Augmentation with Network Interpolation)
- **FakeCLR** (Contrastive Learning with Fake Representations)
- **InsGen** (Instance Generation for Few-Shot Learning)

---

## 🔬 2. Evaluation Methodology (The "Why" and "How")

### The "How"
Our evaluation methodology breaks away from the standard practice of solely analyzing the final, fully-trained model. Instead, we executed a rigorous, automated pipeline running evaluation scripts to calculate **FID, KID, LPIPS, MS-SSIM, Precision, and Recall** on *every single saved training snapshot* across the entire training lifecycle.

### The "Why"
Evaluating per-snapshot allows us to map the temporal evolution of the neural networks. This granular approach is vital for low-shot regimes because it empirically proves:
- Exactly *when* mode collapse begins.
- How rapidly the discriminator overfits on limited datasets.
- The precise trajectory of generative degradation over time, offering insights that a final-snapshot evaluation fundamentally obscures.

---

## ⚙️ 3. The Data Pipeline (The JSON Compilation)

Tracking over 6 complex metrics across hundreds of snapshots generates a massive volume of unstructured raw data. To process this, we built a highly specialized data pipeline:

1. **Extraction:** Raw textual and log outputs from individual evaluation scripts (`calc_metrics.py`, `calc_lpips.py`, etc.) were captured during runtime.
2. **Parsing:** A programmatic parser stripped the logs, isolated the numeric values, and structurally compiled them into normalized **JSON format**.
3. **Ingestion:** This structured JSON compilation served as the critical bridge, allowing our Jupyter Notebooks to seamlessly ingest the data and plot the comparative temporal trajectory graphs. 

---

## 🚨 4. Key Finding: The "FID Illusion"

Through our temporal plotting, we observed a critical anomaly in generative evaluation which we term the **"FID Illusion."**

Our metrics revealed that models operating in low-shot regimes can frequently achieve artificially strong (low) FID scores, appearing highly successful on paper. However, when these FID scores are cross-referenced with heavily degraded **Recall** and **Coverage** metrics, severe **mode collapse** is exposed. The model learns to perfectly replicate a microscopic subset of the dataset (driving down FID) while entirely failing to generate diverse, novel samples.

---

## 📁 5. Repository Structure

This repository has been cleanly refactored and organized into dynamic, relative paths for seamless execution.

```text
Few-Shot-GM-Analysis/
├── src/               # Core architectures and evaluation scripts
│   ├── DANI/
│   ├── FakeCLR/
│   ├── insgen/
│   └── stylegan2-ADA-pytorch/
├── notebooks/         # Jupyter notebooks for JSON ingestion and temporal trajectory plotting
├── samples/           # Selected high-quality output grids demonstrating generation capabilities
├── Few_shot_Generative_Modelling_Analysis.pdf              # PDFs, Reports, and documentation

```

---

## 🌐 6. External Links & Resources

Due to strict file size limits, the final `.pkl` model snapshots are externally hosted.

- 📥 **Model Weights:** [Hugging Face Model Weights](https://huggingface.co/VishyRK/Few-Shot-GM-DC-Weights/tree/main)
- 📄 **Research Report:** *[Placeholder for the final 48-page PDF report]*
this is the readme it gave , is this fine?
