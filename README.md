# 🧠 Resume Ranking using SBERT and Spiking Neural Networks (SNNs)

This project implements an **unsupervised resume-job matching system** using **Spiking Neural Networks (SNNs)**, **Transformer attention**, and **FAISS** for fast similarity search. It's designed to simulate brain-inspired AI for efficiently matching resumes to job descriptions.

## 🚀 Project Highlights

- 🔩 **Spiking Transformer Block** mimics biological neurons using Leaky Integrate-and-Fire (LIF) spiking.
- 🔍 **FAISS Indexing** for fast cosine similarity-based resume retrieval.
- 🧠 **Unsupervised Silhouette Loss** optimizes embeddings without labeled data.
- 📊 **Mean Reciprocal Rank (MRR)** to evaluate top-k relevance.
- 🔥 **Mixed-Precision Training** using PyTorch AMP for speed and lower memory usage.

---

## 🧩 Project Steps

This section explains each major step involved in building and executing this project:

### 🔹 1. **Data Collection**
- Resume PDFs or text files are gathered and stored in the `resumes/` directory.
- Job descriptions are stored in `job_descriptions.csv`.

---

### 🔹 2. **Embedding Generation**
- SBERT are used to convert text into dense vectors.
- Resume and job embeddings are stored in `.pkl` (Pickle) format for quick access.
  - Example: `resume_embeddings.pt`, `job_embeddings.pt`

---

### 🔹 4. **Spike Encoding**
- Dense embeddings are converted into spiking representations using population encoding strategies.
- These spikes simulate biological neuron activity.

---

### 🔹 5. **Spiking Neural Network (SNN) Architecture**
- **Input Size**: 3840 (Concatenated Job + Resume embeddings)
- **Transformer Attention**: Multi-head (8 heads)
- **Layers**:
  - Spiking Transformer
  - Linear → LIF
  - Linear → LIF
- **Output**: Ranked similarity scores for resumes

---

