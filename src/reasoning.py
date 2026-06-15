from typing import Dict, List, Any, Tuple


VECTOR_DB_NAMES = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "chromadb", "chroma",
}

AI_SKILL_KEYWORDS = [
    "embeddings", "faiss", "qdrant", "milvus", "pinecone", "weaviate",
    "rag", "llm", "nlp", "pytorch", "tensorflow", "transformers",
    "fine-tuning", "lora", "machine learning", "deep learning",
    "recommendation systems", "ranking", "search", "mlops",
    "prompt engineering", "openai", "gpt", "bert", "sentence-transformers",
    "semantic search", "vector search", "information retrieval",
]

RETRIEVAL_KEYWORDS = [
    "embeddings", "retrieval", "ranking", "search", "recommendation",
    "vector", "faiss", "pinecone", "qdrant", "milvus", "elasticsearch",
    "opensearch", "semantic search", "dense retrieval", "hybrid search",
]

PRODUCTION_KEYWORDS = [
    "production", "deployed", "shipped", "served", "real users", "at scale",
    "launched", "live system", "production system",
]


def _normalize_skill(name: str) -> str:
    return name.lower().strip()


def _get_verified_ai_skills(skills: List[Dict]) -> List[str]:
    verified = []
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        for kw in AI_SKILL_KEYWORDS:
            if kw in name_lower:
                prof = sk.get("proficiency", "")
                endorsements = sk.get("endorsements", 0)
                duration = sk.get("duration_months", 0)
                if (prof in ("advanced", "expert") and duration >= 12) or endorsements >= 15:
                    verified.append(sk.get("name", ""))
                break
    return verified


def _has_vector_db(skills: List[Dict]) -> bool:
    return any(_normalize_skill(s["name"]) in VECTOR_DB_NAMES for s in skills)


def _has_production_evidence(career: List[Dict]) -> bool:
    for role in career:
        desc = (role.get("description", "") or "").lower()
        if any(kw in desc for kw in PRODUCTION_KEYWORDS):
            return True
    return False


def _has_retrieval_experience(career: List[Dict], skills: List[Dict]) -> bool:
    career_text = " ".join((r.get("description", "") + " " + r.get("title", "")).lower() for r in career)
    skill_text = " ".join(s.get("name", "").lower() for s in skills)
    all_text = career_text + " " + skill_text
    return any(kw in all_text for kw in RETRIEVAL_KEYWORDS)


def _get_product_company_roles(career: List[Dict]) -> List[str]:
    product_indicators = {
        "software", "fintech", "saas", "ai", "ml", "data", "cloud",
        "platform", "product", "startup", "e-commerce", "marketplace",
        "healthtech", "edtech", "biotech", "deeptech",
    }
    companies = []
    for role in career:
        industry = (role.get("industry", "") or "").lower()
        if any(ind in industry for ind in product_indicators):
            comp = role.get("company", "")
            if comp and comp not in companies:
                companies.append(comp)
    return companies


def _get_ai_career_ratio(career: List[Dict]) -> float:
    ai_months = 0
    total_months = 0
    ai_kws = ["ml", "ai ", "machine learning", "deep learning", "nlp",
               "embeddings", "retrieval", "ranking", "search engine",
               "recommendation", "data scien", "data engineer"]
    for role in career:
        months = role.get("duration_months", 0)
        total_months += months
        desc = (role.get("description", "") or "").lower()
        title_r = (role.get("title", "") or "").lower()
        combined = desc + " " + title_r
        for kw in ai_kws:
            if kw in combined:
                ai_months += months
                break
    return ai_months / max(total_months, 1)


def _get_concerns(
    profile: Dict,
    career: List[Dict],
    skills: List[Dict],
    signals: Dict,
    honeypot_info: tuple,
    score: float,
) -> List[str]:
    concerns = []
    years = profile.get("years_of_experience", 0)
    low, high = 5, 9
    if years < low:
        concerns.append(f"below target experience range ({years:.0f}yr vs {low}-{high}yr)")
    elif years > high + 3:
        concerns.append(f"significantly over target experience ({years:.0f}yr)")

    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    preferred = ["pune", "noida"]
    acceptable = ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurgaon", "gurugram"]
    if not any(city in location for city in preferred + acceptable):
        if country != "india":
            concerns.append("non-India location")
        else:
            concerns.append("outside preferred Pune/Noida")

    notice = signals.get("notice_period_days", 60)
    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")

    resp_rate = signals.get("recruiter_response_rate", 0)
    if resp_rate < 0.3:
        concerns.append("low recruiter response rate")

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            from datetime import datetime
            active = datetime.strptime(last_active, "%Y-%m-%d")
            ref = datetime(2026, 6, 15)
            days_since = (ref - active).days
            if days_since > 180:
                concerns.append("inactive for 6+ months")
        except (ValueError, TypeError):
            pass

    if honeypot_info and honeypot_info[0]:
        concerns.append("profile flagged for inconsistencies")

    ai_ratio = _get_ai_career_ratio(career)
    if ai_ratio < 0.3:
        concerns.append("limited AI/ML career history")

    return concerns


def generate_reasoning(
    candidate: Dict,
    scores: Dict[str, float],
    honeypot_info: tuple,
    rank: int = 50,
) -> str:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    score = scores.get("semantic", 0)

    verified_skills = _get_verified_ai_skills(skills)
    has_vdb = _has_vector_db(skills)
    has_prod = _has_production_evidence(career)
    has_retrieval = _has_retrieval_experience(career, skills)
    product_companies = _get_product_company_roles(career)
    ai_ratio = _get_ai_career_ratio(career)
    concerns = _get_concerns(profile, career, skills, signals, honeypot_info, score)

    strengths = []
    if verified_skills:
        skill_str = ", ".join(verified_skills[:3])
        strengths.append(f"verified {skill_str}")
    if has_vdb:
        strengths.append("vector DB production experience")
    if has_retrieval:
        strengths.append("retrieval/ranking systems background")
    if has_prod:
        strengths.append("production deployment evidence")
    if product_companies:
        strengths.append(f"product-company background ({', '.join(product_companies[:2])})")
    if ai_ratio >= 0.5:
        strengths.append(f"{ai_ratio:.0%} of career in AI/ML")

    signals_note = []
    if signals.get("open_to_work_flag"):
        signals_note.append("actively open to work")
    resp_rate = signals.get("recruiter_response_rate", 0)
    if resp_rate >= 0.7:
        signals_note.append(f"highly responsive ({resp_rate:.0%})")
    notice = signals.get("notice_period_days", 60)
    if notice <= 15:
        signals_note.append(f"{notice}-day notice period")

    if rank <= 10:
        parts = []
        if strengths:
            parts.append(f"{title} ({years:.0f}yr) with {strengths[0]}")
            if len(strengths) > 1:
                parts.append(strengths[1])
        if has_prod and has_retrieval:
            parts.append("directly matches JD's core requirement of shipped retrieval systems")
        elif has_prod:
            parts.append("has shipped systems to real users as JD requires")
        if signals_note:
            parts.append(signals_note[0])
        if concerns:
            parts.append(f"minor gap: {concerns[0]}")
    elif rank <= 30:
        parts = []
        if strengths:
            parts.append(f"{title} with {', '.join(strengths[:2])}")
        if has_retrieval:
            parts.append("aligns with JD's retrieval/ranking focus")
        if signals_note:
            parts.append(signals_note[0])
        if concerns:
            parts.append(f"concern: {concerns[0]}")
    elif rank <= 60:
        parts = []
        parts.append(f"{title} ({years:.0f}yr)")
        if verified_skills:
            parts.append(f"skills include {', '.join(verified_skills[:2])}")
        if not has_prod:
            parts.append("limited production deployment evidence")
        if concerns:
            parts.append(f"gap: {concerns[0]}")
    else:
        parts = []
        parts.append(f"{title} with {years:.0f}yr experience")
        if verified_skills:
            parts.append(f"some relevant skills ({', '.join(verified_skills[:2])})")
        else:
            parts.append("limited verified AI/ML skills for this role")
        if concerns:
            parts.append(f"concerns: {', '.join(concerns[:2])}")
        if not has_retrieval:
            parts.append("no retrieval/ranking systems experience")

    if honeypot_info and honeypot_info[0]:
        reasons = honeypot_info[2][:2]
        parts.append(f"[FLAGGED: {'; '.join(reasons)}]")

    return "; ".join(parts[:5])


def generate_reasoning_batch(
    candidates: List[Dict],
    score_details: List[Dict[str, float]],
    honeypot_infos: List[tuple],
    ranks: List[int] = None,
) -> List[str]:
    if ranks is None:
        ranks = [50] * len(candidates)
    reasonings = []
    for cand, scores, honeypot, rank in zip(candidates, score_details, honeypot_infos, ranks):
        reasoning = generate_reasoning(cand, scores, honeypot, rank)
        reasonings.append(reasoning)
    return reasonings
