# Redrob Intelligent Candidate Discovery & Ranking

AI-powered candidate ranking system for the Redrob Data & AI Challenge. Ranks 100K candidates against a Senior AI Engineer job description using a hybrid multi-signal scoring approach that mirrors how a great recruiter evaluates candidates.

## Approach

### Architecture

The system uses a **three-layer scoring architecture**:

1. **Semantic Layer (TF-IDF + cosine similarity)** — Captures overall profile-to-JD relevance beyond keyword matching using 50K TF-IDF features with bigrams
2. **Domain Intelligence Layer (rule-based scoring)** — Evaluates career trajectory, trust-weighted skills, and role alignment
3. **Behavioral Layer (signal scoring)** — Weights engagement, availability, and recruiter signals

### Why This Architecture

The JD explicitly warns: *"The right answer is NOT find candidates whose skills section contains the most AI keywords."* A pure keyword/embedding approach would be gamed by candidates who stuff AI keywords into unrelated profiles (the dataset contains these traps).

Our system addresses this by:

- **Career trajectory > skill lists**: We score career descriptions (what they actually did) higher than skill tags (what they claim to know)
- **Trust-weighted skills**: Skills are weighted by proficiency level, endorsement count, and usage duration — catching keyword stuffers who list "expert" with 0 endorsements
- **Anti-disqualification logic**: Penalizes consulting-only careers, CV/speech-only backgrounds without NLP/IR, and candidates with no production ML evidence
- **Honeypot detection**: Identifies ~2000 candidates with impossible profiles (duration mismatches, expert skills with 0 months usage, salary contradictions)
- **Behavioral signals**: Down-weights inactive candidates, non-responders, and those with long notice periods

### Scoring Components

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| TF-IDF Semantic Similarity | 22% | Overall profile-to-JD text relevance |
| Title/Career Alignment | 20% | AI/ML career trajectory strength |
| Weighted AI Skills | 12% | Trust-weighted core AI skill match |
| Career Description Match | 8% | AI keywords in actual job descriptions |
| Retrieval Skills | 7% | Vector DB, search, retrieval experience |
| Vector DB Skills | 5% | Specific vector database experience |
| LLM Skills | 3% | LLM/fine-tuning experience |
| AI Career Ratio | 5% | % of career spent in AI/ML roles |
| Experience Match | 4% | 5-9 year sweet spot |
| Product Company Score | 3% | Product vs consulting background |
| Location Score | 2% | Pune/Noida > India > International |
| Work Mode Score | 2% | Hybrid preference match |
| Behavioral Score | 3% | Availability, responsiveness, engagement |
| GitHub Activity | 1% | Open-source contribution signal |
| Assessment Score | 1% | Platform skill assessment results |
| Education Score | 1% | Relevant field + institution tier |
| Python Bonus | 5% | Has Python in skill set |
| Production Description | 10% | Career descriptions mention production/shipped |

### Disqualification Penalties

- **Honeypots**: Score reduced by up to 80% (duration mismatches, impossible skill profiles)
- **Consulting-only careers**: 40% penalty if no product-company experience and weak AI title alignment
- **CV/Speech-only without NLP/IR**: 30% penalty for candidates whose expertise is in computer vision/speech without NLP exposure
- **No production ML**: 30% penalty for candidates with no evidence of shipping ML systems

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Pre-compute (run once)
```bash
python rank.py --precompute --candidates ./candidates.jsonl --artifacts ./artifacts
```

### Step 2: Rank (must complete within 5 min on CPU)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Validate
```bash
python validate_submission.py submission.csv
```

## Compute Constraints

- **Pre-computation**: ~2 minutes (TF-IDF vectorization of 100K candidates)
- **Ranking**: ~45 seconds on CPU (well within 5-minute limit)
- **Memory**: ~2GB peak (TF-IDF sparse matrix + candidate data)
- **No GPU required**
- **No network calls during ranking**

## File Structure

```
├── rank.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── validate_submission.py     # Submission validator
├── submission.csv             # Output ranking (top 100)
├── submission_metadata.yaml   # Team & approach metadata
├── README.md                  # This file
├── LICENSE                    # MIT License
├── src/
│   ├── __init__.py
│   ├── scorer.py             # Multi-signal scoring engine
│   ├── precompute.py         # TF-IDF pre-computation
│   └── reasoning.py          # Natural language reasoning generator
└── artifacts/                 # Pre-computed TF-IDF matrices
    ├── tfidf_candidates.npz
    ├── tfidf_jd.npz
    ├── tfidf_ids.json
    ├── tfidf_vectorizer.pkl
    └── jd_similarities.npy
```

## License

MIT
