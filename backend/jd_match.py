"""Job-description matching. Free, local, no API key.

Two engines:
  - tfidf:      scikit-learn TF-IDF + cosine similarity
  - semantic:   sentence-transformers all-MiniLM-L6-v2
"""
from __future__ import annotations

import re
from typing import Optional


def _clean(s: str) -> str:
    s = re.sub(r"[^\w\s+#./-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def match_tfidf(resume_text: str, jd_text: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [_clean(resume_text), _clean(jd_text)]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          stop_words="english", max_features=5000)
    m = vec.fit_transform(corpus)
    score = float(cosine_similarity(m[0], m[1])[0][0])

    feature_names = vec.get_feature_names_out()
    jd_vec = m[1].toarray()[0]
    resume_vec = m[0].toarray()[0]

    jd_terms = {feature_names[i]: jd_vec[i] for i in jd_vec.nonzero()[0]}
    resume_terms = {feature_names[i]: resume_vec[i] for i in resume_vec.nonzero()[0]}

    missing = sorted(
        [(t, w) for t, w in jd_terms.items() if t not in resume_terms],
        key=lambda x: -x[1],
    )[:20]
    matched = sorted(
        [(t, w) for t, w in jd_terms.items() if t in resume_terms],
        key=lambda x: -x[1],
    )[:20]

    return {
        "score": round(score, 4),
        "score_pct": round(score * 100, 1),
        "engine": "tfidf",
        "matched_terms": [t for t, _ in matched],
        "missing_terms": [t for t, _ in missing],
    }


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _MODEL = False
    return _MODEL if _MODEL is not False else None


def match_semantic(resume_text: str, jd_text: str) -> Optional[dict]:
    model = _get_model()
    if not model:
        return None
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        embs = model.encode([resume_text[:4000], jd_text[:4000]],
                            normalize_embeddings=True)
        score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        return {"score": round(score, 4),
                "score_pct": round(score * 100, 1),
                "engine": "semantic"}
    except Exception:
        return None


def match_jd(resume_text: str, jd_text: str, prefer: str = "auto") -> dict:
    if not jd_text or len(jd_text.strip()) < 40:
        return {"score": 0.0, "score_pct": 0.0, "engine": "none",
                "error": "Job description too short to match."}

    semantic = None
    if prefer in ("semantic", "auto"):
        semantic = match_semantic(resume_text, jd_text)

    tfidf = match_tfidf(resume_text, jd_text)

    if semantic:
        blended = 0.6 * semantic["score"] + 0.4 * tfidf["score"]
        return {
            **tfidf,
            "score": round(blended, 4),
            "score_pct": round(blended * 100, 1),
            "engine": "blended",
            "semantic_score_pct": semantic["score_pct"],
            "tfidf_score_pct": tfidf["score_pct"],
        }

    return tfidf