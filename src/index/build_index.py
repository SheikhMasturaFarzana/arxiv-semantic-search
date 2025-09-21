import os
import json
import faiss
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("index/faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def merge_and_deduplicate(limit: int = None) -> pd.DataFrame:
    """
    Merge all processed JSONL files, deduplicate by arxiv_id, and return as DataFrame.
    If limit is given, truncate to that many rows (for testing).
    """
    files = list(PROCESSED_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError("No processed files found in data/processed/")

    records = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["arxiv_id"], inplace=True)

    if limit:
        df = df.head(limit)

    # Save merged version back to processed/
    merged_path = PROCESSED_DIR / "merged.jsonl"
    df.to_json(merged_path, orient="records", lines=True, force_ascii=False)

    # Delete old files (except merged.jsonl)
    for file in files:
        if file.name != "merged.jsonl":
            file.unlink()

    return df


def build_index(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", limit: int = None):
    """
    Build FAISS index on abstracts.
    """
    print("[index] Merging processed files...")
    df = merge_and_deduplicate(limit=limit)

    print(f"[index] Loaded {len(df)} documents after deduplication.")

    # Extract texts for embeddings
    texts = df["abstract"].fillna("").tolist()

    print("[index] Loading embedding model...")
    model = SentenceTransformer(model_name)

    print("[index] Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Save embeddings
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    # Save metadata (lightweight version)
    metadata_fields = [
        "arxiv_id", "title", "abstract", "summary",
        "authors", "categories", "affiliations",
        "keywords", "language", "published_date", "url_pdf"
    ]
    df_meta = df[metadata_fields].copy()
    df_meta.to_json(INDEX_DIR / "metadata.jsonl", orient="records", lines=True, force_ascii=False)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (works since we normalized)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    print(f"[index] Done. {len(df)} docs indexed.")
    print(f"[index] Saved files:")
    print(f" - {INDEX_DIR / 'embeddings.npy'}")
    print(f" - {INDEX_DIR / 'metadata.jsonl'}")
    print(f" - {INDEX_DIR / 'faiss.index'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from processed data")
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="SentenceTransformer model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs (for testing)")

    args = parser.parse_args()
    build_index(model_name=args.model, limit=args.limit)