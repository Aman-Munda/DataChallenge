#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranker
Ranks 100K candidates against a Senior AI Engineer job description.
Produces a top-100 CSV with candidate_id, rank, score, reasoning.

Usage:
    # Step 1: Pre-compute (run once, can take longer than 5 min)
    python rank.py --precompute --candidates ./candidates.jsonl --artifacts ./artifacts

    # Step 2: Rank (must complete within 5 min on CPU)
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np


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


def load_candidates(path: str) -> list:
    candidates = []
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates: list, jd_sim: np.ndarray, sim_ids: list) -> list:
    from src.scorer import (
        compute_behavioral_score,
        compute_experience_match,
        compute_location_score,
        compute_product_company_score,
        compute_skill_overlap,
        compute_title_alignment,
        compute_weighted_skill_score,
        compute_work_mode_score,
        detect_honeypot,
        has_career_description_match,
        compute_assessment_score,
        CORE_AI_SKILLS,
        VECTOR_DB_SKILLS,
        RETRIEVAL_SKILLS,
        LLM_SKILLS,
    )

    id_to_sim = {cid: i for i, cid in enumerate(sim_ids)}

    print("Scoring all candidates...")
    results = []
    honeypot_count = 0

    for i, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        education = cand.get("education", [])
        signals = cand.get("redrob_signals", {})

        idx = id_to_sim.get(cid)
        sem_score = float(jd_sim[idx]) if idx is not None else 0.0

        is_honeypot, honeypot_penalty, honeypot_reasons = detect_honeypot(cand)
        if is_honeypot:
            honeypot_count += 1

        title_score = compute_title_alignment(profile, career)

        ai_skill_score = compute_weighted_skill_score(skills, CORE_AI_SKILLS)
        vector_db_score = compute_weighted_skill_score(skills, VECTOR_DB_SKILLS)
        retrieval_score = compute_weighted_skill_score(skills, RETRIEVAL_SKILLS)
        llm_score = compute_weighted_skill_score(skills, LLM_SKILLS)

        career_desc_ai = has_career_description_match(career, [
            "embeddings", "vector", "retrieval", "ranking", "search",
            "recommendation", "machine learning", "deep learning", "nlp",
            "natural language", "llm", "fine-tuning", "transformer",
            "pytorch", "tensorflow", "ml pipeline", "feature engineering",
            "model training", "model serving", "inference",
        ])

        exp_score = compute_experience_match(profile.get("years_of_experience", 0))
        location_score = compute_location_score(profile)
        work_mode_score = compute_work_mode_score(signals)
        behavioral_score = compute_behavioral_score(signals)
        product_score = compute_product_company_score(career)
        assessment_score = compute_assessment_score(signals, CORE_AI_SKILLS)

        ai_career_months = 0
        total_months = 0
        for role in career:
            months = role.get("duration_months", 0)
            total_months += months
            desc = (role.get("description", "") or "").lower()
            title_r = (role.get("title", "") or "").lower()
            combined = desc + " " + title_r
            ai_kws = ["ml", "ai ", "machine learning", "deep learning", "nlp",
                       "embeddings", "retrieval", "ranking", "search engine",
                       "recommendation", "data scien", "data engineer"]
            for kw in ai_kws:
                if kw in combined:
                    ai_career_months += months
                    break
        ai_career_ratio = ai_career_months / max(total_months, 1)

        github_score = signals.get("github_activity_score", -1)
        github_normalized = min(github_score / 50.0, 1.0) if github_score >= 0 else 0.0

        has_python = any("python" in s.get("name", "").lower() for s in skills)
        python_bonus = 0.05 if has_python else 0.0

        has_prod_desc = False
        for role in career:
            desc = (role.get("description", "") or "").lower()
            if any(kw in desc for kw in ["production", "deployed", "shipped", "served", "real users", "at scale"]):
                has_prod_desc = True
                break
        prod_desc_bonus = 0.1 if has_prod_desc else 0.0

        education_score = 0.0
        for edu in education:
            field = (edu.get("field_of_study", "") or "").lower()
            tier = edu.get("tier", "unknown")
            if any(kw in field for kw in ["computer science", "data science", "artificial intelligence",
                                          "machine learning", "software engineering", "information technology"]):
                education_score += 0.3
            if tier == "tier_1":
                education_score += 0.3
            elif tier == "tier_2":
                education_score += 0.2
            elif tier == "tier_3":
                education_score += 0.1
        education_score = min(education_score, 1.0)

        raw_score = (
            0.22 * sem_score +
            0.20 * title_score +
            0.12 * ai_skill_score +
            0.08 * career_desc_ai +
            0.07 * retrieval_score +
            0.05 * vector_db_score +
            0.03 * llm_score +
            0.05 * ai_career_ratio +
            0.04 * exp_score +
            0.03 * product_score +
            0.02 * location_score +
            0.02 * work_mode_score +
            0.03 * behavioral_score +
            0.01 * github_normalized +
            0.01 * assessment_score +
            0.01 * education_score +
            python_bonus +
            prod_desc_bonus
        )

        if is_honeypot:
            raw_score *= (1.0 - honeypot_penalty * 0.8)

        consulting_only = True
        for role in career:
            company = (role.get("company", "") or "").lower()
            if not any(c in company for c in ["tcs", "infosys", "wipro", "accenture",
                                               "cognizant", "capgemini", "hcl", "tech mahindra",
                                               "mindtree", "lti", "mphasis"]):
                consulting_only = False
                break
        if consulting_only and title_score < 0.3:
            raw_score *= 0.6

        career_text = " ".join((r.get("description", "") + " " + r.get("title", "")).lower() for r in career)
        skill_text = " ".join(s.get("name", "").lower() for s in skills)
        all_text = career_text + " " + skill_text
        has_ir_nlp = any(kw in all_text for kw in ["nlp", "natural language", "text", "ir ",
                                                     "search", "retrieval", "ranking", "embeddings"])
        has_cv_speech = any(kw in all_text for kw in ["computer vision", "image class", "object detection",
                                                       "speech", "audio", "yolo", "opencv", "tts"])
        if has_cv_speech and not has_ir_nlp and ai_career_ratio < 0.3:
            raw_score *= 0.7

        no_prod_ml = ai_career_ratio <= 0.2 and career_desc_ai <= 0.2 and not has_prod_desc
        if no_prod_ml and title_score < 0.3:
            raw_score *= 0.7

        score = max(0.0, min(1.0, raw_score))

        results.append({
            "candidate_id": cid,
            "score": score,
            "is_honeypot": is_honeypot,
            "honeypot_info": (is_honeypot, honeypot_penalty, honeypot_reasons),
            "score_details": {
                "semantic": sem_score,
                "title": title_score,
                "ai_skills": ai_skill_score,
                "career_desc": career_desc_ai,
                "retrieval": retrieval_score,
                "vector_db": vector_db_score,
                "llm": llm_score,
                "ai_career_ratio": ai_career_ratio,
                "experience": exp_score,
                "product": product_score,
                "location": location_score,
                "behavioral": behavioral_score,
            },
            "candidate": cand,
        })

        if (i + 1) % 10000 == 0:
            print(f"  Scored {i + 1}/{len(candidates)}")

    print(f"Found {honeypot_count} potential honeypots")
    for r in results:
        r["score"] = round(r["score"], 6)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return results


def generate_submissions_csv(results: list, output_path: str, top_n: int = 100):
    from src.reasoning import generate_reasoning

    top = results[:top_n]
    for i, r in enumerate(top):
        r["rank"] = i + 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top:
            reasoning = generate_reasoning(
                r["candidate"],
                r["score_details"],
                r["honeypot_info"],
                rank=r["rank"],
            )
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.6f}",
                reasoning,
            ])
    print(f"Written top {top_n} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--artifacts", default="./artifacts", help="Pre-computed artifacts directory")
    parser.add_argument("--precompute", action="store_true", help="Run pre-computation step")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top candidates to rank")
    args = parser.parse_args()

    if args.precompute:
        from src.precompute import precompute_tfidf
        os.makedirs(args.artifacts, exist_ok=True)
        print("=== Pre-computation Phase ===")
        precompute_tfidf(args.candidates, args.artifacts)
        print("Pre-computation complete.")
        return

    start_time = time.time()

    print("=== Loading Artifacts ===")
    sim_path = os.path.join(args.artifacts, "jd_similarities.npy")
    ids_path = os.path.join(args.artifacts, "tfidf_ids.json")

    if not os.path.exists(sim_path):
        print(f"ERROR: Pre-computed similarities not found at {sim_path}")
        print("Run with --precompute first to generate artifacts.")
        sys.exit(1)

    jd_sim = np.load(sim_path)
    with open(ids_path, "r") as f:
        sim_ids = json.load(f)
    print(f"Loaded similarities: {jd_sim.shape}")

    print("=== Loading Candidates ===")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates")

    print("=== Ranking ===")
    results = rank_candidates(candidates, jd_sim, sim_ids)

    print("=== Generating Output ===")
    generate_submissions_csv(results, args.out, args.top_n)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Top 5 candidates:")
    for r in results[:5]:
        cand = r["candidate"]
        print(f"  #{r.get('rank', '?')}: {cand['candidate_id']} - "
              f"{cand['profile']['current_title']} ({cand['profile']['years_of_experience']}y) - "
              f"Score: {r['score']:.6f}")


if __name__ == "__main__":
    main()
