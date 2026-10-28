# 🎬 ShotList Video RAG & Visual Search Engine

A multimodal Video Retrieval-Augmented Generation (RAG) system built from scratch. It allows users to search through thousands of video shots and exact timestamps using natural language queries, powered by **SigLIP** embeddings and **ChromaDB**.

![Python](https://img.shields.io/badge/Python-3.10%23-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-red)
![SigLIP](https://img.shields.io/badge/Model-SigLIP-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)

---

## 🚀 Key Features

* **Advanced Scene Detection:** Automatically splits videos into meaningful semantic shots using `PySceneDetect`.
* **Smart Subsampling:** Intelligently samples keyframes based on shot duration to avoid visual redundancy while capturing key actions.
* **Multimodal Embeddings (SigLIP):** Uses Google's SigLIP model to map text queries and image frames into a shared vector space for high-precision visual matching.
* **Filename Sanitization Pipeline:** Automatically cleans and normalizes metadata and file paths, avoiding path resolution bugs caused by special characters or spaces.
* **Rigorous Evaluation Suite:** Evaluates retrieval performance using standard Information Retrieval metrics (**Hit Rate @K** and **Mean Reciprocal Rank**).

---

## 📂 Project Structure

```text
shotlist_rag/
│
├─  assets/                    # Screenshots showing the frontend and different search results
├── data/
│   ├── videos_ingested/       # Raw video files
│   └── extracted_shots/       # Processed keyframes (auto-sanitized paths)
├── databases/
|   ├── chroma_db/             # Vector database to store embeddings
|   └── sqlite/                # Sql database to store file hashes and avoid uploaded duplicates
├─  scripts/                   # Script to create a sample video to ingest and test the video processor
├── services/
│   ├── video_processor.py     # Scene detection, SigLIP encoding & frame cleaning
│   ├── db_manager.py          # ChromaDB connection and vector search
│     ├── db_sqlite.py         # Sqlite connection
│   └── task_manager.py        #  
├── tests/
│   ├── golden_dataset.json    # Benchmark queries with expected frame, title and timestamp
│   ├── test_main.py           # Test the api endpoints and functionalities with Mocks 
│   ├── test_videoprocessor.py # Frame extraction and video processing
│   └── evaluate_rag.py        # Evaluation script (Hit Rate & MRR)
├──config.py                   # Database and data paths
├── schemas.py                 # Pydantic schemas 
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── index.html                 # Frontend with HTML,CSS and Javascripts
└── main.py

```

---

## 📈 Evaluation & Performance

Unlike traditional text-based RAGs (where LLM text-generation metrics like RAGAS apply), Video RAG systems require **Information Retrieval (IR)** metrics to measure spatial and temporal accuracy. 

The evaluation suite tests the system against a custom *Golden Dataset* measuring exact shot matches within a temporal tolerance window.

### Benchmark Results (Sample Run)
* **Total Test Cases:** 5 complex visual queries across ~5,000 indexed shots.
* **Hit Rate @1 (Exact match in 1st position):** `80.0%`
* **Hit Rate @3 (Found within Top 3 results):** `100.0%`
* **MRR (Mean Reciprocal Rank):** `90.0%`

*(Detailed results are automatically exported to `video_rag_evaluation_report.csv` after each test run).*

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Antonio-Fernandez-AiEng/shotlist-video-rag.git
cd shotlist_rag
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run with Docker (Alternative)
If you prefer containerized execution:
```bash
docker compose up --build
```

---

## 🧪 Running the Evaluation Suite

To test the accuracy of the vector search engine against the Golden Dataset:
```bash
python tests/evaluate_rag.py
```

---

## 💡 Engineering Highlights
* **Why IR Metrics over LLM Metrics?** Evaluated via *Hit Rate @K* and *MRR* rather than text-generation frameworks, ensuring true mathematical validation of visual grounding without unnecessary LLM overhead or NaN errors on blank text contexts.
* **Robust File Handling:** Implemented a regex-based filename sanitizer to prevent OS-level `FileNotFoundError` exceptions when handling raw movie titles containing special characters (`#`, `|`, brackets, etc.).