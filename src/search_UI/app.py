# src/ui/app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
INDEX_DIR = Path("index/")
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"

# ---------- Constants ----------
POOL_K = 200      # pool size
MIN_SIM = 0.40    # similarity threshold
MAX_RESULTS = 20  # display at most this many results

# ---------- Caching ----------
@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    df = pd.read_json(METADATA_PATH, lines=True)
    for col in ["authors", "categories", "affiliations", "keywords"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
    if "language" not in df.columns:
        df["language"] = ""
    if "published_date" not in df.columns:
        df["published_date"] = None
    df["year"] = pd.to_datetime(df["published_date"], errors="coerce").dt.year
    return df

@st.cache_resource(show_spinner=False)
def load_index_and_model():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return index, model

# ---------- Search ----------
def faiss_search(query: str, index, model):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(q_emb, POOL_K)
    return sims[0], idxs[0]

def list_contains_any(lst, selected):
    if not selected:
        return True
    if not isinstance(lst, (list, tuple, set)):
        return False
    s = set(str(x) for x in lst)
    return any(x in s for x in selected)

def render_result(row: pd.Series, score: float):
    title = (row.get("title", "(untitled)")).replace("\n", " ").strip()
    url_pdf = row.get("url_pdf") or "#"
    summary = row.get("summary") or ""
    abstract = row.get("abstract", "") or ""
    keywords = row.get("keywords") or []

    # Title
    st.markdown(f"### [{title}]({url_pdf})")

    # Summary
    if summary:
        st.markdown(f"**Summary:** {summary}")

    # Abstract
    if abstract:
        st.markdown("**Abstract:**")
        st.markdown(abstract)

    # Keywords
    if keywords:
        styled_keywords = ", ".join([f"<span class='kw'>{k}</span>" for k in keywords])
        st.markdown(f"**Keywords:** {styled_keywords}", unsafe_allow_html=True)

    # Similarity score
    st.markdown(f"<div class='score'>similarity: {score:.3f}</div>", unsafe_allow_html=True)
    st.divider()

# ---------- UI ----------
st.set_page_config(page_title="ArXiv Search", page_icon="🔎", layout="wide")

# custom CSS
st.markdown("""
<style>
.kw {
  color: #1a73e8;
  font-style: italic;
  margin-right: 6px;
}
.score {
  color: #6b7280;
  font-size: 0.85rem;
  margin-top: 6px;
}
section[data-testid="stSidebar"] { background-color: #f6f9ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin-top:0'>ArXiv Search</h2>", unsafe_allow_html=True)

df_meta = load_metadata()
index, model = load_index_and_model()

query = st.text_input("Type your query: ", value="", placeholder="e.g., retrieval-augmented generation for healthcare")

if query.strip():
    with st.spinner("Searching..."):
        sims, idxs = faiss_search(query.strip(), index, model)
        valid = [(int(i), float(s)) for i, s in zip(idxs, sims) if i >= 0 and s >= float(MIN_SIM)]

        if valid:
            res_df = df_meta.iloc[[v[0] for v in valid]].copy()
            res_df["__score__"] = [v[1] for v in valid]
        else:
            res_df = pd.DataFrame(columns=list(df_meta.columns) + ["__score__"])

    with st.sidebar:
        st.header("Filters")

        def collect_unique(list_series):
            vals = set()
            for x in list_series:
                if isinstance(x, (list, tuple, set)):
                    vals.update(str(v) for v in x)
            return sorted(vals)

        author_opts = collect_unique(res_df.get("authors", pd.Series([], dtype=object)))
        category_opts = collect_unique(res_df.get("categories", pd.Series([], dtype=object)))
        affil_opts = collect_unique(res_df.get("affiliations", pd.Series([], dtype=object)))
        lang_opts = sorted(set(str(x) for x in res_df.get("language", pd.Series([], dtype=object)) if pd.notna(x) and str(x)))

        min_year = int(res_df["year"].min()) if not res_df.empty and pd.notna(res_df["year"]).any() else 1990
        max_year = int(res_df["year"].max()) if not res_df.empty and pd.notna(res_df["year"]).any() else 2030
        if min_year == max_year:
            min_year = max_year - 1  # fallback for slider

        sel_authors = st.multiselect("Authors", author_opts)
        sel_categories = st.multiselect("Categories", category_opts)
        sel_affils = st.multiselect("Affiliations", affil_opts)
        sel_langs = st.multiselect("Language", lang_opts)
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))

    # Apply filters
    if not res_df.empty:
        def pass_filters(r: pd.Series) -> bool:
            if not (year_range[0] <= (int(r["year"]) if pd.notna(r["year"]) else year_range[0]) <= year_range[1]):
                return False
            if sel_langs and (str(r.get("language") or "") not in set(sel_langs)):
                return False
            if not list_contains_any(r.get("authors"), sel_authors):
                return False
            if not list_contains_any(r.get("categories"), sel_categories):
                return False
            if not list_contains_any(r.get("affiliations"), sel_affils):
                return False
            return True

        filtered = res_df[res_df.apply(pass_filters, axis=1)].copy()
        filtered = filtered.sort_values("__score__", ascending=False).head(MAX_RESULTS)
    else:
        filtered = pd.DataFrame(columns=list(df_meta.columns) + ["__score__"])

    if filtered.empty:
        st.info("No results found.")
    else:
        st.caption(f"Showing {len(filtered)} results (min similarity {MIN_SIM:.2f})")
        for _, row in filtered.iterrows():
            render_result(row, score=row["__score__"])
else:
    with st.sidebar:
        st.header("Filters")