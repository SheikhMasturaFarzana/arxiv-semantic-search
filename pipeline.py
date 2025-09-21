# pipeline.py
import argparse
import subprocess
from src.datasource.arxiv_crawler import crawl_arxiv_to_jsonl
from src.preprocess import preprocess
from src.index.build_index import build_index


def cmd_crawl(args: argparse.Namespace) -> None:
    count, outpath = crawl_arxiv_to_jsonl(
        categories=args.categories,
        max_results=args.max_results,
        query=args.query,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )
    print(f"[pipeline] crawl finished → {count} docs saved at {outpath}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    preprocess.main()


def cmd_embed(args: argparse.Namespace) -> None:
    print("[pipeline] embed not implemented yet")


def cmd_index(args: argparse.Namespace) -> None:
    build_index(model_name=args.model, limit=args.limit)


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch Streamlit app (UI)."""
    app_path = "src/search_UI/app.py"
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except FileNotFoundError:
        print("[error] Streamlit is not installed. Run `pip install streamlit` first.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArXiv semantic search pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # crawl
    pc = sub.add_parser("crawl", help="Crawl latest results and save raw JSONL")
    pc.add_argument("--categories", nargs="*", default=["cs.AI", "cs.CL", "cs.LG"])
    pc.add_argument("--max-results", type=int, default=5000)
    pc.add_argument("--query", type=str, default=None)
    pc.add_argument("--verbose", action="store_true")
    pc.add_argument("--no-progress", action="store_true")
    pc.set_defaults(func=cmd_crawl)

    # preprocess
    sub.add_parser("preprocess", help="Process raw → canonical").set_defaults(func=cmd_preprocess)

    # embed (stub for now)
    sub.add_parser("embed", help="Generate embeddings").set_defaults(func=cmd_embed)

    # index
    pi = sub.add_parser("index", help="Build FAISS index")
    pi.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model to use",
    )
    pi.add_argument("--limit", type=int, default=None, help="Limit number of docs (for testing)")
    pi.set_defaults(func=cmd_index)

    # serve
    sub.add_parser("serve", help="Launch Streamlit app").set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
