import json
import gzip
import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def build_candidate_text(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    parts = []

    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    current = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if current:
        parts.append(f"{current} {company}")

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    for role in career[:4]:
        title = role.get("title", "")
        comp = role.get("company", "")
        desc = role.get("description", "")
        if title:
            parts.append(f"{title} {comp} {desc}")

    skill_strs = []
    for sk in skills:
        name = sk.get("name", "")
        prof = sk.get("proficiency", "")
        if name:
            skill_strs.append(f"{name}")
    if skill_strs:
        parts.append(" ".join(skill_strs))

    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        if degree:
            parts.append(f"{degree} {field} {inst}")

    return " ".join(parts)


JD_TEXT = (
    "Senior AI Engineer Redrob AI Series AI native talent intelligence platform "
    "Own intelligence layer ranking retrieval matching systems "
    "Production experience embeddings retrieval systems sentence transformers OpenAI embeddings BGE "
    "vector databases Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS "
    "strong Python evaluation frameworks ranking NDCG MRR MAP "
    "LLM fine tuning LoRA QLoRA PEFT learning to rank HR tech recruiting "
    "distributed systems large scale inference optimization open source contributions "
    "5 9 years experience applied ML AI product companies "
    "shipped ranking search recommendation system real users meaningful scale "
    "embeddings dense retrieval hybrid search evaluation offline online "
    "NLP natural language processing information retrieval IR "
    "machine learning deep learning PyTorch TensorFlow transformers "
    "data pipelines feature engineering model serving MLOps "
    "Pune Noida India hybrid work mode "
    "Python code quality software engineering "
    "recommendation systems collaborative filtering content based "
    "BM25 semantic search dense passage retrieval "
    "RAG retrieval augmented generation "
    "A B testing experimentation frameworks "
)


def precompute_tfidf(candidates_path: str, output_dir: str):
    print("Loading candidates...")
    candidates = []
    if candidates_path.endswith(".gz"):
        with gzip.open(candidates_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates")

    print("Building candidate texts...")
    texts = []
    ids = []
    for i, cand in enumerate(candidates):
        text = build_candidate_text(cand)
        texts.append(text)
        ids.append(cand["candidate_id"])
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i + 1}/{len(candidates)}")

    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )

    all_texts = texts + [JD_TEXT]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    candidate_tfidf = tfidf_matrix[:-1]
    jd_tfidf = tfidf_matrix[-1:]

    print(f"TF-IDF matrix shape: {candidate_tfidf.shape}")

    os.makedirs(output_dir, exist_ok=True)

    from scipy.sparse import save_npz
    save_npz(os.path.join(output_dir, "tfidf_candidates.npz"), candidate_tfidf)
    save_npz(os.path.join(output_dir, "tfidf_jd.npz"), jd_tfidf)

    with open(os.path.join(output_dir, "tfidf_ids.json"), "w") as f:
        json.dump(ids, f)

    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved TF-IDF artifacts to {output_dir}")

    print("Computing JD similarities...")
    jd_sim = cosine_similarity(candidate_tfidf, jd_tfidf).flatten()
    np.save(os.path.join(output_dir, "jd_similarities.npy"), jd_sim)
    print(f"Saved JD similarities: shape={jd_sim.shape}, min={jd_sim.min():.4f}, max={jd_sim.max():.4f}")

    return candidate_tfidf, ids, jd_sim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", default="./artifacts")
    args = parser.parse_args()

    start = time.time()
    precompute_tfidf(args.candidates, args.output_dir)
    print(f"Total pre-computation: {time.time() - start:.1f}s")
