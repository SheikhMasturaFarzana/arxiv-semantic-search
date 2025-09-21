import os
import json
import time
import requests
import feedparser
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from tqdm.auto import tqdm

ARXIV_API = "http://export.arxiv.org/api/query"
HEADERS = {
    "User-Agent": "curl/8.6.0",
    "Accept": "application/atom+xml",
}

# ----------------------------
# Helpers
# ----------------------------

def build_query(categories: List[str], query: Optional[str]) -> str:
    """Build a safe arXiv search_query: (cat:cs.AI OR cat:cs.LG) AND your terms"""
    cats = [c.strip() for c in (categories or []) if c.strip()]
    cat_clause = " OR ".join(f"cat:{c}" for c in cats)
    cat_clause = f"({cat_clause})" if cat_clause else ""
    if query and cat_clause:
        return f"{cat_clause} AND {query}"
    return query or cat_clause or "all"

def entry_to_rawdoc(e: Dict, crawl_date: Optional[str] = None) -> Dict:
    authors = []
    for a in e.get("authors", []) or []:
        name = getattr(a, "name", None)
        if name:
            authors.append(name)

    categories = []
    for t in e.get("tags", []) or []:
        term = t.get("term")
        if term:
            categories.append(term)

    prim = ""
    prim_cat = e.get("arxiv_primary_category") or {}
    if isinstance(prim_cat, dict):
        prim = prim_cat.get("term", "") or ""

    pdf_url = None
    for l in e.get("links", []) or []:
        if l.get("type") == "application/pdf":
            pdf_url = l.get("href")
            break

    return {
        "arxiv_id": (e.get("id", "") or "").rsplit("/", 1)[-1],
        "title": (e.get("title", "") or "").strip(),
        "abstract": (e.get("summary", "") or "").strip(),
        "authors": authors,
        "primary_category": prim,
        "categories": categories,
        "published": e.get("published"),
        "updated": e.get("updated"),
        "url_abs": e.get("id"),
        "url_pdf": pdf_url,
        "comment": e.get("arxiv_comment"),
        "journal_ref": e.get("arxiv_journal_ref"),
        "doi": e.get("arxiv_doi"),
        "crawl_date": crawl_date,
    }

def _write_jsonl(path: str, docs: List[Dict], show_progress: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(tqdm(docs, disable=not show_progress, desc="Writing JSONL", unit="doc")):
            f.write(json.dumps(d, ensure_ascii=False))
            if i < len(docs) - 1:
                f.write("\n")

# ----------------------------
# Core fetcher (curl-equivalent)
# ----------------------------

def _fetch_results(query: str, max_results: int, show_progress: bool, verbose: bool=False) -> List[Dict]:
    """Fetch up to max_results using properly encoded params and paging."""
    ARXIV_API = "https://export.arxiv.org/api/query"
    HEADERS = {"User-Agent": "curl/8.6.0", "Accept": "application/atom+xml"}

    results: List[Dict] = []
    fetched = 0
    pbar = tqdm(total=max_results, disable=not show_progress, desc="Fetching", unit="doc")

    try:
        while fetched < max_results:
            ask = min(2000, max_results - fetched)  # request up to 2000
            params = {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": fetched,
                "max_results": ask,
            }
            resp = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=60)
            if verbose:
                print(f"[fetch] GET {resp.url} -> {resp.status_code}")
            if resp.status_code != 200:
                break

            feed = feedparser.parse(resp.text)
            entries = feed.entries or []

            if verbose and fetched == 0:
                perpage = feed.feed.get("opensearch_itemsperpage")
                total = feed.feed.get("opensearch_totalresults")
                print(f"[fetch] itemsPerPage={perpage}, total={total}, got={len(entries)}")

            if not entries:
                # No more results
                break

            results.extend(entries)
            fetched += len(entries)
            pbar.update(len(entries))

            time.sleep(4)  # ToU: >= 3s between requests

    finally:
        pbar.close()

    return results

# ----------------------------
# Main crawler
# ----------------------------

def crawl_arxiv_to_jsonl(
    categories: List[str],
    max_results: int = 5000,
    query: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> Tuple[int, str]:
    """Crawl arXiv and save the latest max_results results to JSONL."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cats_name = "-".join(c.replace(".", "_") for c in (categories or [])) or "all"
    outdir = os.path.join("data", "raw")
    outpath = os.path.join(outdir, f"{timestamp}_{cats_name}.jsonl")
    crawl_date = timestamp[:8]

    full_query = build_query(categories, query)

    if verbose:
        print(f"[crawl] query: {full_query}")
        print(f"[crawl] max_results={max_results}")
        print(f"[crawl] output: {outpath}")

    entries = _fetch_results(full_query, max_results, show_progress, verbose=verbose)

    # convert + de-dupe by arxiv_id (safety)
    rawdocs = [entry_to_rawdoc(e, crawl_date=crawl_date) for e in tqdm(entries, disable=not show_progress, desc="Converting", unit="doc")]
    seen, deduped = set(), []
    for d in rawdocs:
        aid = d.get("arxiv_id")
        if aid and aid not in seen:
            seen.add(aid)
            deduped.append(d)

    _write_jsonl(outpath, deduped, show_progress)

    if verbose:
        print(f"[crawl] wrote {len(deduped)} records to {outpath}")

    return len(deduped), os.path.abspath(outpath)
