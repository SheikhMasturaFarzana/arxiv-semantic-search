import os
import json
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from langdetect import detect

# Import helpers
from src.preprocess.pdf_parser import extract_first_page_text
from src.preprocess.llm_extract import extract_metadata as llm_extract_metadata

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ARCHIVE_DIR = Path("data/archive")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PATH = Path("src/preprocess/prompt.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()

PROMPT_PATH = Path("src/preprocess/prompt_nometadata.txt")
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    NOMETADATA_PROMPT = f.read()

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def extract_metadata(doc: Dict) -> Dict:
    # Base fields we already have
    metadata = {
        "arxiv_id": doc.get("arxiv_id"),
        "title": doc.get("title", "").strip(),
        "abstract": doc.get("abstract", "").strip(),
        "authors": doc.get("authors", []),
        "categories": doc.get("categories", []),
        "primary_category": doc.get("primary_category", ""),
        "published_date": doc.get("published"),
        "url_pdf": doc.get("url_pdf"),
        "crawl_date": doc.get("crawl_date"),
        "language": detect_language(doc.get("abstract", "")),

        # New: first-page raw text from PDF
        "pdf_raw": "",

        # Placeholders / later filled
        "summary": None,
        "affiliations": [],
        "keywords": [],
        "locations": [],
    }

    # --- Step 1: extract first-page PDF text ---
    url_pdf = doc.get("url_pdf")
    if url_pdf:
        try:
            text = extract_first_page_text(url_pdf)
            metadata["pdf_raw"] = text
        except Exception as e:
            print(f"[warn] PDF parse failed for {doc.get('arxiv_id')}: {e}")

    # --- Step 2: call LLM parser if pdf_raw is non-empty ---
    if metadata["pdf_raw"]:
        try:
            result = llm_extract_metadata(metadata["pdf_raw"], BASE_PROMPT)
            metadata["affiliations"] = result.get("affiliations", [])
            metadata["keywords"] = result.get("keywords", [])
            metadata["summary"] = result.get("summary", "")
        except Exception as e:
            print(f"[warn] LLM parse failed for {doc.get('arxiv_id')}: {e}")

        return metadata
    else:
        # Fallback: no pdf_raw available, use abstract with simpler prompt
        try:
            result = llm_extract_metadata(metadata["abstract"], NOMETADATA_PROMPT)
            metadata["keywords"] = result.get("keywords", [])
            metadata["summary"] = result.get("summary", "")
        except Exception as e:
            print(f"[warn] LLM parse (no pdf) failed for {doc.get('arxiv_id')}: {e}")
        return metadata


def process_raw_file(filepath: Path):
    with filepath.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    enriched = [extract_metadata(doc) for doc in tqdm(records, desc=f"Processing {filepath.name}")]

    outpath = PROCESSED_DIR / filepath.name
    with outpath.open("w", encoding="utf-8") as out:
        for doc in enriched:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    shutil.move(str(filepath), ARCHIVE_DIR / filepath.name)


def main():
    raw_files = list(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print("No files to process in data/raw/")
        return

    for file in raw_files:
        try:
            process_raw_file(file)
            print(f"[✓] Processed and archived: {file.name}")
        except Exception as e:
            print(f"[!] Failed to process {file.name}: {e}")


if __name__ == "__main__":
    main()