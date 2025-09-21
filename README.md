
---

# ArXiv Semantic Search

## Overview

This project implements a semantic search engine over arXiv abstracts using **FAISS** as the vector database.
The pipeline covers:

1. **Crawling** arXiv metadata.
2. **Preprocessing** raw dumps into clean, enriched metadata (LLM extraction + PDF parsing).
3. **Indexing** documents into FAISS with embeddings from `sentence-transformers`.
4. **UI**: A Streamlit front end to query semantically, view abstracts, and filter by metadata.

---

## Project Structure

```
arxiv-search/
│
├── data/
│   ├── raw/         # Raw JSONL dumps (arXiv API)
│   ├── processed/   # Cleaned metadata, enriched with LLM outputs
│   ├── archive/     # Archived raw files after processing
│   └── pdf_cache/   # Downloaded PDFs (cached locally)
│
├── src/
│   ├── preprocess/
│   │   ├── pdf_parser.py         # First-page text extractor (PDF → text)
│   │   ├── llm_extract.py        # JSON LLM extractor (affiliations, keywords, summary)
│   │   ├── preprocess.py         # End-to-end metadata processor
│   │   ├── prompt.txt            # Base LLM prompt
│   │   └── prompt_nometadata.py  # Fallback prompt (no affiliations)
│   │
│   ├── index/
│   │   └── build_index.py        # Build FAISS index + metadata files
│   │
│   └── search_UI/
│       └── app.py                # Streamlit front end
│
├── pipeline.py     # CLI: crawl → preprocess → index → serve
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a conda env:

```bash
conda create -n arxivsearch python=3.10 -y
conda activate arxivsearch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Key requirements:

* `arxiv`
* `tqdm`
* `langdetect`
* `requests`
* `PyMuPDF`
* `sentence-transformers`
* `faiss-cpu`
* `streamlit`
* `openai` 

---

## Usage

### 1. Crawl arXiv

Fetch raw metadata and save JSONL in `data/raw/`.

```bash
python pipeline.py crawl --categories cs.AI --max-results 1000 --verbose
```

### 2. Preprocess

Normalize metadata, detect language, parse first page of PDFs, and enrich with LLM outputs.

```bash
python pipeline.py preprocess
```
Preprocessed files go to `data/processed/`, and raw files are archived.

### 3. Build Index

Generate embeddings (from abstracts) and build FAISS index.

```bash
python pipeline.py index
```
Default model - `-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This creates inside `index/faiss/`:

* `faiss.index` — FAISS vector index
* `embeddings.npy` — embeddings
* `metadata.jsonl` — cleaned metadata aligned with the index

### 4. Run Search UI

Launch the Streamlit app:

```bash
python pipeline.py serve
```
Open in browser (default: `http://localhost:8501`).

---

## Search UI

* **Query**: Search by semantic meaning of abstracts.
* **Results**: Show Title (linked to PDF), Summary, Abstract, Keywords, and similarity score.
* **Filters**: On the left sidebar, refine by Authors, Categories, Affiliations, Language, and Year range.

---

## Notes

* **LLM Enrichment**: Requires an OpenAI API key (`OPENAI_API_KEY` in your env). Used in `llm_extract.py`.
* **PDF Parsing**: Only the first page is downloaded and cached in `data/pdf_cache/`.
* **Indexing**: Always deduplicates and merges all files in `data/processed/` before building the index.

---