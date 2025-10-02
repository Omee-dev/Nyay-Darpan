# retrieval.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import pandas as pd
# ------------------ Load Data ------------------

# Case data (assumed to be loaded in your main script)
case_data = pd.read_csv('./Data/output_ss3.csv')  # your deduplicated case data
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # adjust if using another model

# Citations
with open("./Data/Sectional_Digest_Chunks.json", "r", encoding="utf-8") as f:
    citation_data = json.load(f)

citation_embeddings = np.load("./Data/Sectional_Digest_Chunks_embeddings.npy")  # ndarray

# Sectoral
with open("./Data/Sectoral_Digest_Chunks.json", "r", encoding="utf-8") as f:
    sectoral_data = json.load(f)

npz_file = np.load("./Data/Sectoral_Digest_Chunks_embeddings.npz", allow_pickle=True)

sectoral_embeddings = npz_file["embeddings"]
sectoral_sectors = npz_file["sectors"]
sectoral_data = npz_file["data"].tolist()

# ------------------ BM25 Precompute ------------------
citation_corpus = [word_tokenize(c['content'].lower()) for c in citation_data]
bm25_citation = BM25Okapi(citation_corpus)

sectoral_corpus = [word_tokenize(c['Content'].lower()) for c in sectoral_data]
bm25_sectoral = BM25Okapi(sectoral_corpus)

# ------------------ Sector Map ------------------
sector_map = {
    "113": ["AIRLINES SECTOR"],
    "101": ["BANKING SECTOR"],
    "111": ["DRUGS AND COSMETICS"],
    "108": ["E-COMMERCE SECTOR"],
    "120": ["EDUCATION SECTOR"],
    "116": ["ELECTRICITY SECTOR"],
    "118": ["FOOD SECTOR", "HOTELS & RESTAURANTS"],
    "102": ["INSURANCE SECTOR"],
    "112": ["MEDICAL SECTOR"],
    "128": ["POSTAL SECTOR"],
    "114": ["RAILWAYS SECTOR"],
    "115": ["REAL ESTATE SECTOR"],
    "109": ["TELECOM SECTOR"],
    "999": ["OTHERS"]
}

def find_similar_cases(overview_text, sector_code=999):
    """Find top 5 similar cases based on embeddings"""
    global case_data
    if case_data is None:
        raise ValueError("case_data must be loaded in main app and assigned here.")

    try:
        sector_code = int(sector_code)
    except:
        sector_code = 999

    matched_cases = case_data[case_data['Sector Code'] == sector_code]
    if matched_cases.empty:
        matched_cases = case_data

    overview_embedding = embed_model.encode(overview_text, convert_to_tensor=True)
    brief_texts = matched_cases['Brief'].dropna().astype(str).tolist()
    brief_embeddings = embed_model.encode(brief_texts, convert_to_tensor=True)

    similarities = util.cos_sim(overview_embedding, brief_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:5].cpu().numpy().tolist()
    top_cases = matched_cases.iloc[top_indices][
        ['Judgement name', 'Citation', 'No', 'Brief','Link']
    ]

    results = []
    for _, row in top_cases.iterrows():
        results.append({
            "judgement": row['Judgement name'],
            "citation": row['Citation'],
            "no": row['No'],
            "brief": row['Brief'],
            'link': str(row['Link']) if str(row['Link']).startswith('https') else 'NA'
        })

    return results


# ------------------ Citations ------------------
citation_embeddings = np.load(
    r"./Data/Sectional_Digest_Chunks_embeddings.npy", allow_pickle=True
)

# Ensure it's 2D
if citation_embeddings.ndim == 1:
    citation_embeddings = np.vstack(citation_embeddings)

# ------------------ Retrieval ------------------
def get_top5_citations(query_text, model, k=5):
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    sims = (citation_embeddings @ query_embedding.T).squeeze()  # fast dot product
    top_indices = sims.argsort()[-k:][::-1]
    return [citation_data[i] for i in top_indices]


def get_top5_citations_bm25(query_text, k=5):
    """BM25 citation retrieval"""
    query_tokens = word_tokenize(query_text.lower())
    scores = bm25_citation.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [citation_data[i] for i in top_indices]

def get_top5_sectoral_chunks(query_text, sector_code, k=5):
    """Embedding-based retrieval restricted to a given sector code"""
    sector_names = sector_map.get(str(sector_code), [])
    print(sector_names, sector_code)
    if not sector_names:
        return []

    # Filter by sector
    sector_indices = [i for i, s in enumerate(sectoral_data) if s["Sector"] in sector_names]
    if not sector_indices:
        return []

    sector_embeddings = sectoral_embeddings[sector_indices]
    sector_subset = [sectoral_data[i] for i in sector_indices]

    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)
    sims = cosine_similarity(query_embedding, sector_embeddings)[0]

    top_indices = sims.argsort()[-k:][::-1]
    results = []
    for i in top_indices:
        s = sector_subset[i]
        results.append({
            "title": s.get("Title", "Untitled"),
            "sector": s.get("Sector", "Unknown"),
            "citation": s.get("Citation", s.get("citation_source", "")),
            "content": s.get("Content", s.get("content", ""))
        })
    return results


def get_top5_sectoral_chunks_bm25(query_text, sector_code, k=5):
    """BM25 retrieval restricted to a given sector code"""
    sector_names = sector_map.get(str(sector_code), [])
    if not sector_names:
        return []

    sector_indices = [i for i, s in enumerate(sectoral_data) if s["Sector"] in sector_names]
    if not sector_indices:
        return []

    sector_subset = [sectoral_data[i] for i in sector_indices]
    sector_corpus = [sectoral_corpus[i] for i in sector_indices]

    bm25_local = BM25Okapi(sector_corpus)
    query_tokens = word_tokenize(query_text.lower())
    scores = bm25_local.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for i in top_indices:
        s = sector_subset[i]
        results.append({
            "title": s.get("Title", "Untitled"),
            "sector": s.get("Sector", "Unknown"),
            "citation": s.get("Citation", s.get("citation_source", "")),
            "content": s.get("Content", s.get("content", ""))
        })
    return results
