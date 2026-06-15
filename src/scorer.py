import re
import math
from typing import Dict, List, Any, Optional, Tuple


JD_ROLE = {
    "title": "Senior AI Engineer",
    "company": "Redrob AI",
    "company_stage": "Series A AI-native talent intelligence platform",
    "location_preferred": ["Pune", "Noida"],
    "location_acceptable": ["Hyderabad", "Mumbai", "Delhi NCR", "Bangalore", "Bengaluru", "Gurgaon", "Gurugram"],
    "experience_range": (5, 9),
    "work_mode": "hybrid",
}

CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "bert", "transformers", "hugging face transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "rag", "retrieval augmented generation", "vector databases", "vector search",
    "recommendation systems", "ranking", "search", "information retrieval",
    "nlp", "natural language processing", "llm", "large language models",
    "fine-tuning llms", "lora", "qlora", "peft", "fine-tuning",
    "pytorch", "tensorflow", "machine learning", "deep learning",
    "mlops", "model deployment", "model serving",
    "prompt engineering", "langchain",
    "xgboost", "gradient boosting",
    "feature engineering", "data pipelines",
    "ndcg", "mrr", "map", "evaluation frameworks",
    "openai", "gpt", "claude",
    "python",
}

VECTOR_DB_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "chromadb", "chroma", "annoy",
}

RETRIEVAL_SKILLS = {
    "rag", "retrieval augmented generation", "vector search", "vector databases",
    "embeddings", "sentence-transformers", "bert", "hugging face transformers",
    "bm25", "hybrid search", "semantic search", "dense retrieval",
    "recommendation systems", "ranking", "search", "information retrieval",
}

LLM_SKILLS = {
    "llm", "large language models", "fine-tuning llms", "lora", "qlora", "peft",
    "prompt engineering", "langchain", "openai", "gpt", "claude",
    "fine-tuning", "transformer", "attention mechanism",
}

ML_FOUNDATION_SKILLS = {
    "machine learning", "deep learning", "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "xgboost", "gradient boosting",
    "nlp", "natural language processing", "computer vision",
    "reinforcement learning", "bayesian", "statistical modeling",
}

PYTHON_ECOSYSTEM = {
    "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "pandas", "numpy", "fastapi", "flask", "django", "bentoml",
    "mlflow", "wandb", "weights & biases", "kubeflow",
}

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "lti", "mphasis",
    "deloitte", "pwc", "ey", "kpmg", "mckinsey", "bcg", "bain",
}

PRODUCT_INDICATORS = {
    "software", "fintech", "saas", "ai", "ml", "data", "cloud",
    "platform", "product", "startup", "e-commerce", "marketplace",
    "healthtech", "edtech", "biotech", "deeptech",
}


def normalize_skill(name: str) -> str:
    return name.lower().strip()


def compute_skill_overlap(candidate_skills: List[Dict], target_set: set) -> Tuple[float, List[str]]:
    matched = []
    for sk in candidate_skills:
        name = normalize_skill(sk["name"])
        if name in target_set:
            matched.append(name)
        for target in target_set:
            if target in name or name in target:
                if target not in matched:
                    matched.append(target)
    return len(matched) / max(len(target_set), 1), matched


def compute_weighted_skill_score(candidate_skills: List[Dict], target_set: set) -> float:
    if not candidate_skills:
        return 0.0

    prof_weights = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
    total_score = 0.0
    max_possible = 0.0

    for sk in candidate_skills:
        name = normalize_skill(sk["name"])
        is_match = False
        for target in target_set:
            if target in name or name in target:
                is_match = True
                break

        if is_match:
            prof = prof_weights.get(sk.get("proficiency", "beginner"), 0.2)
            endorsements = min(sk.get("endorsements", 0) / 20.0, 1.0)
            duration = min(sk.get("duration_months", 0) / 36.0, 1.0)
            trust = 0.4 * prof + 0.3 * endorsements + 0.3 * duration
            total_score += trust

        max_possible += 1.0

    return min(total_score / max(len(target_set), 1), 1.0)


def has_career_description_match(career_history: List[Dict], keywords: List[str]) -> float:
    if not career_history:
        return 0.0

    total_months = 0
    matched_months = 0

    for role in career_history:
        desc = (role.get("description", "") or "").lower()
        title = (role.get("title", "") or "").lower()
        combined = desc + " " + title
        months = role.get("duration_months", 0)
        total_months += months

        for kw in keywords:
            if kw.lower() in combined:
                matched_months += months
                break

    return matched_months / max(total_months, 1)


def compute_experience_match(years: float) -> float:
    low, high = JD_ROLE["experience_range"]
    if low <= years <= high:
        return 1.0
    elif years < low:
        return max(0.0, 1.0 - (low - years) / low)
    else:
        excess = years - high
        return max(0.0, 1.0 - excess / (high * 2))


def compute_location_score(profile: Dict) -> float:
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()

    for city in JD_ROLE["location_preferred"]:
        if city.lower() in location:
            return 1.0

    for city in JD_ROLE["location_acceptable"]:
        if city.lower() in location:
            return 0.85

    if country == "india":
        return 0.6
    elif country in ("usa", "uk", "canada", "australia", "uae", "singapore"):
        return 0.3
    return 0.1


def compute_work_mode_score(signals: Dict) -> float:
    mode = signals.get("preferred_work_mode", "")
    if mode == "hybrid":
        return 1.0
    elif mode == "flexible":
        return 0.9
    elif mode == "onsite":
        return 0.7
    elif mode == "remote":
        return 0.5
    return 0.5


def compute_behavioral_score(signals: Dict) -> float:
    score = 0.0

    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work:
        score += 0.2

    recruiter_resp = signals.get("recruiter_response_rate", 0)
    score += 0.15 * min(recruiter_resp / 0.5, 1.0)

    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            from datetime import datetime
            active = datetime.strptime(last_active, "%Y-%m-%d")
            ref = datetime(2026, 6, 15)
            days_since = (ref - active).days
            if days_since <= 30:
                score += 0.15
            elif days_since <= 90:
                score += 0.1
            elif days_since <= 180:
                score += 0.05
        except (ValueError, TypeError):
            pass

    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.1
    elif notice <= 90:
        score += 0.05

    profile_views = signals.get("profile_views_received_30d", 0)
    saved = signals.get("saved_by_recruiters_30d", 0)
    engagement = min(profile_views / 50, 1.0) * 0.05 + min(saved / 10, 1.0) * 0.05
    score += engagement

    interview_rate = signals.get("interview_completion_rate", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0:
        score += 0.1 * interview_rate + 0.1 * offer_rate
    else:
        score += 0.1 * interview_rate

    return min(score, 1.0)


def detect_honeypot(candidate: Dict) -> Tuple[bool, float, List[str]]:
    reasons = []
    penalty = 0.0

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    years_exp = profile.get("years_of_experience", 0)

    for role in career:
        start = role.get("start_date", "")
        end = role.get("end_date")
        months = role.get("duration_months", 0)
        if start and months > 0:
            try:
                from datetime import datetime
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime(2026, 6, 15)
                actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                if actual_months > 0 and abs(months - actual_months) > 6:
                    reasons.append(f"Duration mismatch: stated {months}m vs actual ~{actual_months}m")
                    penalty += 0.3
            except (ValueError, TypeError):
                pass

    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if total_career_months > 0:
        stated_months = years_exp * 12
        if abs(total_career_months - stated_months) > 36:
            reasons.append(f"Career total {total_career_months}m vs stated {stated_months}m")
            penalty += 0.2

    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    zero_expert = [s for s in expert_skills if s.get("duration_months", 0) == 0]
    if len(zero_expert) >= 2:
        reasons.append(f"{len(zero_expert)} expert skills with 0 months usage")
        penalty += 0.3

    low_endorse_expert = [s for s in expert_skills if s.get("endorsements", 0) == 0 and s.get("duration_months", 0) < 6]
    if len(low_endorse_expert) >= 3:
        reasons.append(f"{len(low_endorse_expert)} expert skills with no endorsements and <6mo")
        penalty += 0.2

    skill_names = [normalize_skill(s["name"]) for s in skills]
    ai_keywords = ["nlp", "llm", "machine learning", "deep learning", "pytorch", "tensorflow",
                   "transformers", "bert", "gpt", "fine-tuning", "rag", "embeddings",
                   "faiss", "qdrant", "milvus", "openai", "prompt engineering"]
    ai_count = sum(1 for kw in ai_keywords if any(kw in sn for sn in skill_names))

    title = (profile.get("current_title", "") or "").lower()
    non_ai_titles = ["marketing", "accountant", "sales", "customer support",
                     "operations manager", "hr manager", "content writer",
                     "graphic designer", "civil engineer", "mechanical engineer"]
    has_non_ai_title = any(t in title for t in non_ai_titles)

    if ai_count >= 6 and has_non_ai_title:
        reasons.append(f"Keyword stuffing: {ai_count} AI skills but title is '{profile.get('current_title')}'")
        penalty += 0.4

    for edu in education:
        start_y = edu.get("start_year", 0)
        end_y = edu.get("end_year", 0)
        if start_y > 0 and end_y > 0 and end_y < start_y:
            reasons.append(f"Education end year {end_y} before start year {start_y}")
            penalty += 0.3

    max_salary = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    min_salary = signals.get("expected_salary_range_inr_lpa", {}).get("min", 0)
    if max_salary > 0 and min_salary > 0 and min_salary > max_salary:
        reasons.append(f"Salary min {min_salary} > max {max_salary}")
        penalty += 0.2

    is_honeypot = penalty >= 0.3
    return is_honeypot, min(penalty, 1.0), reasons


def compute_title_alignment(profile: Dict, career: List[Dict]) -> float:
    current_title = (profile.get("current_title", "") or "").lower()
    headline = (profile.get("headline", "") or "").lower()
    summary = (profile.get("summary", "") or "").lower()

    ai_engineer_signals = [
        "ai engineer", "ml engineer", "machine learning engineer",
        "data scientist", "data engineer", "backend engineer",
        "software engineer", "full stack", "nlp engineer",
        "research engineer", "applied scientist", "mlops",
        "senior engineer", "staff engineer", "principal engineer",
    ]

    non_ai_signals = [
        "marketing manager", "accountant", "sales executive",
        "customer support", "operations manager", "hr manager",
        "content writer", "graphic designer", "civil engineer",
        "mechanical engineer", "business analyst", "project manager",
    ]

    ai_title_score = 0.0
    combined = current_title + " " + headline
    for sig in ai_engineer_signals:
        if sig in combined:
            ai_title_score = 1.0
            break

    for sig in non_ai_signals:
        if sig in combined:
            ai_title_score = max(0.0, ai_title_score - 0.5)
            break

    career_ai_months = 0
    total_months = 0
    for role in career:
        months = role.get("duration_months", 0)
        total_months += months
        role_title = (role.get("title", "") or "").lower()
        role_desc = (role.get("description", "") or "").lower()
        combined_role = role_title + " " + role_desc

        for sig in ai_engineer_signals:
            if sig in combined_role:
                career_ai_months += months
                break

    career_ratio = career_ai_months / max(total_months, 1)

    summary_ai_keywords = ["embeddings", "retrieval", "ranking", "vector", "ml", "ai",
                           "machine learning", "deep learning", "nlp", "llm",
                           "fine-tuning", "pytorch", "tensorflow", "transformers"]
    summary_hits = sum(1 for kw in summary_ai_keywords if kw in summary)
    summary_score = min(summary_hits / 5, 1.0)

    return 0.35 * ai_title_score + 0.40 * career_ratio + 0.25 * summary_score


def compute_product_company_score(career: List[Dict]) -> float:
    product_months = 0
    total_months = 0

    for role in career:
        months = role.get("duration_months", 0)
        total_months += months
        industry = (role.get("industry", "") or "").lower()
        company = (role.get("company", "") or "").lower()
        desc = (role.get("description", "") or "").lower()

        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_product = (
            any(ind in industry for ind in PRODUCT_INDICATORS) or
            any(p in desc for p in ["product company", "product team", "built", "shipped", "launched"])
        )

        if is_product and not is_consulting:
            product_months += months

    return product_months / max(total_months, 1)


def compute_assessment_score(signals: Dict, target_skills: set) -> float:
    assessments = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.0

    relevant_scores = []
    for skill, score in assessments.items():
        skill_lower = skill.lower()
        for target in target_skills:
            if target in skill_lower or skill_lower in target:
                relevant_scores.append(score / 100.0)
                break

    if not relevant_scores:
        return 0.0

    return sum(relevant_scores) / len(relevant_scores)
