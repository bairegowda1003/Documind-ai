# 🧠 DocuMind AI

> **AI-Powered Academic Knowledge Assistant**  
> Retrieval-Augmented Generation (RAG) System  
> MCA Prompt Engineering Assignment — Chanakya University

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![LLaMA](https://img.shields.io/badge/LLaMA_3-8B_Instruct-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What is DocuMind AI?

DocuMind AI is a production-quality **RAG (Retrieval-Augmented Generation)** system that lets you upload any PDF or TXT document and ask questions about it. The AI answers **strictly from your uploaded documents** — no hallucinations, no guessing.

Built as a Prompt Engineering assignment demonstrating the full RAG pipeline:

```
User Query → Embed Query → Vector Search → Retrieve Chunks → LLaMA 3 → Grounded Answer
```

---

## ✨ Features

- 📄 Upload PDF and TXT documents
- 🔍 Semantic similarity search using `all-MiniLM-L6-v2`
- 🧠 Grounded answers using LLaMA 3 8B Instruct via OpenRouter
- 💾 Persistent ChromaDB vector store
- 📚 Source attribution — see exactly which chunks were used
- 💬 Multi-turn chat with history
- 🎨 Premium Streamlit UI with dark mode toggle
- ⚡ Cached models for fast loading

---

## 🏗️ Architecture

```
rag_project/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .gitignore
├── README.md
│
├── rag/
│   ├── pdf_loader.py       ← Document loading & text extraction
│   ├── chunker.py          ← Overlapping text chunking
│   ├── embeddings.py       ← Sentence-transformer embeddings
│   ├── vector_store.py     ← ChromaDB vector database
│   ├── retriever.py        ← Semantic similarity retrieval
│   ├── llm.py              ← OpenRouter API caller
│   └── rag_pipeline.py     ← Orchestrates full RAG pipeline
│
└── assets/
    └── styles.css          ← Premium UI stylesheet
```

### RAG Pipeline (5 Stages)

| Stage | Module | Description |
|-------|--------|-------------|
| **1. Ingestion** | `pdf_loader.py` | Extract clean text from PDF/TXT |
| **2. Chunking** | `chunker.py` | Split into 500-char overlapping chunks |
| **3. Embedding** | `embeddings.py` | Convert text to 384-dim vectors |
| **4. Storage** | `vector_store.py` | Persist in ChromaDB with cosine similarity |
| **5. Generation** | `rag_pipeline.py` | Retrieve + augment + generate answer |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/documind-ai.git
cd documind-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install downloads the embedding model (~90 MB). This only happens once.

### 4. Set up your API key

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your key:
```
OPENROUTER_API_KEY=your_key_here
```

Get a **free API key** at [openrouter.ai/keys](https://openrouter.ai/keys) — no credit card required for free tier.

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Usage

1. **Upload documents** using the sidebar file uploader (PDF or TXT)
2. Wait for indexing to complete (shown in sidebar)
3. **Type a question** in the chat bar at the bottom
4. View the **grounded answer** and expand **Retrieved Sources** to see evidence

---

## 🔧 Configuration

| Parameter | Default | Location |
|-----------|---------|----------|
| Chunk size | 500 chars | `rag/chunker.py` |
| Chunk overlap | 50 chars | `rag/chunker.py` |
| Top-K retrieval | 5 chunks | `rag/rag_pipeline.py` |
| LLM model | `meta-llama/llama-3-8b-instruct` | `rag/llm.py` |
| Embedding model | `all-MiniLM-L6-v2` | `rag/embeddings.py` |
| Temperature | 0.2 | `rag/llm.py` |

---

## 📸 Screenshots

<img width="1869" height="918" alt="docMIND" src="https://github.com/user-attachments/assets/caf9d4b0-2bd3-491e-a4be-9dcd80c49fd1" />


## 🤖 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend + Backend | Streamlit |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM API | OpenRouter (LLaMA 3 8B Instruct) |
| PDF Parsing | PyPDF |
| Environment | python-dotenv |

---

## 📚 References

- Lewis et al. (2020) — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Course: Prompt Engineering — Chanakya University, School of Engineering
- Instructor: Mr. Deepak B
- Student: Baire Gowda

---

## 📄 License

MIT License — free to use for academic and personal projects.
