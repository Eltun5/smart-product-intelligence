# Smart Product Intelligence

End-to-end deep learning system for an Amazon product catalog (category: **Toys & Games**).
Capstone project — Khazar University, Hands-on Deep Learning course.

---

## 📌 Project Overview

This project applies the full spectrum of modern deep learning to a single real-world
problem: understanding and assisting buyers of toys on Amazon. A single coherent dataset
(Amazon Reviews 2023, Toys & Games category) is attacked with progressively more powerful
tools, from a logistic-regression baseline through transformers and diffusion models,
then integrated into one working assistant.

Six architectures are combined:

| Architecture | Used for | Milestone |
|---|---|---|
| Multilayer Perceptron (MLP) | Rating-band classification from tabular features | M1 |
| CNN + Transfer Learning (MobileNetV2) | Image-based subcategory classification | M2 |
| TF-IDF + Sentence Embeddings | Sentiment classification & semantic search | M3 |
| Transformer (DistilBERT fine-tuning) | High-accuracy sentiment classification | M4 |
| LLM (Llama 3.1 8B) + RAG + T5 fine-tune | Review summarization and grounded Q&A | M5 |
| Stable Diffusion v1.5 | Lifestyle product image generation | M6 |

---

## 📊 Results Summary

| Milestone | Task | Baseline | Deep Model | Improvement |
|---|---|---|---|---|
| **M1 — Rating band** | 3-class tabular | LogReg F1 = 0.372 | MLP F1 = **0.428** | **+15.0%** |
| **M2 — Subcategory** | 6-class from images | CNN-scratch (overfits) | MobileNetV2 acc = **0.631** | Transfer learning wins |
| **M3 — Sentiment** | Binary positive/negative | TF-IDF F1 = **0.864** | Embeddings F1 = 0.826 | TF-IDF wins (honest) |
| **M4 — Sentiment** | Same task, transformer | TF-IDF F1 = 0.864 | DistilBERT F1 = **0.891** | +2.6 pp |

### Key honest findings

1. **TF-IDF beats sentence embeddings on sentiment** — embeddings are 7× slower and slightly less accurate.
2. **DistilBERT accuracy drops with review length** — from ~95% on short reviews to ~88% on 500+ char reviews.
3. **Transfer learning matters in CV** — MobileNetV2 reaches ~66% accuracy with fewer parameters than from-scratch CNN.
4. **Diffusion fails on brands** — Funko, Playmobil, licensed characters are not reproduced.

---

## 📁 Repository Structure
smart-product-intelligence/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_tabular_mlp.ipynb
│   ├── 02_vision_cnn_transfer.ipynb
│   ├── 03_text_embeddings.ipynb
│   ├── 04_transformers.ipynb
│   ├── 05_llm_rag_finetune.ipynb
│   └── 06_diffusion.ipynb
├── src/                  # Reusable modules
├── app/                  # Milestone 7 Streamlit demo
├── figures/              # 9 PNGs
└── report/
└── final_report.pdf
---

## 🚀 Setup

```bash
git clone https://github.com/Eltun5/smart-product-intelligence.git
cd smart-product-intelligence
pip install -r requirements.txt
```

### Dataset

**Amazon Reviews 2023** (McAuley Lab) — Toys & Games category:
- 15,000 unique products
- 75,000 reviews
- ~8,000 cached product images
- Product-level split (no leakage): Train 12,000 / Val 1,500 / Test 1,500

---

## 🛠️ Technology Stack

| Layer | Tool |
|---|---|
| Deep learning framework | Keras / TensorFlow 2.20 |
| Transformer fine-tuning | Hugging Face transformers + PyTorch |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API — Llama 3.1 8B Instant |
| Diffusion | diffusers — Stable Diffusion v1.5 |
| Compute | Google Colab (NVIDIA T4 GPU) |

---

## ✅ Brief Compliance

| Requirement | Status |
|---|---|
| Single category | ✅ Toys & Games |
| 10K-20K products, 30K-80K reviews | ✅ 15K / 75K |
| 5K-10K cached images | ✅ 7,996 |
| Product-level split (no leakage) | ✅ intersection = 0 |
| Baseline before deep model | ✅ all milestones |
| Macro-F1 (rating imbalance) | ✅ all classification |
| Honest error analysis | ✅ confusion matrices, per-class F1 |
| Reproducible via requirements.txt | ✅ |
| All 8 milestones (M0-M7) | ✅ |

---

## 👤 Author

Eltun — Khazar University, 2025
Capstone for the *Hands-on Deep Learning* course
