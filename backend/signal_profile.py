"""Signal -- evidence-based profile summarisation.

Two modes:
  - heuristic:  no API key, regex + keyword patterns
  - llm:        requires GROQ_API_KEY (free, no credit card)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalArchetype:
    name: str
    score: float
    evidence: list = field(default_factory=list)


@dataclass
class SignalProfile:
    summary: str
    archetypes: list
    notable: list
    gaps: list
    confidence: str


_ARCHETYPE_VERBS = {
    "Builder": [
        r"\bbuilt\b", r"\bshipped\b", r"\bimplemented\b", r"\bdesigned\b",
        r"\bcreated\b", r"\bdeveloped\b", r"\bprototyped\b", r"\bwrote\b",
        r"\bengineered\b", r"\bconstructed\b",
    ],
    "Scaler": [
        r"\bscaled\b", r"\boptimized\b", r"\bimproved\b", r"\breduced\b",
        r"\bcut\b", r"\baccelerated\b", r"\bstreamlined\b", r"\bmigrated\b",
        r"\brefactor(?:ed|ing)\b",
    ],
    "Initiator": [
        r"\bfounded\b", r"\blaunched\b", r"\binitiated\b", r"\bestablished\b",
        r"\bstarted\b", r"\bled\b", r"\bintroduced\b", r"\bpioneered\b",
        r"\bspearheaded\b", r"\bdrove\b",
    ],
    "Operator": [
        r"\bmanaged\b", r"\bcoordinated\b", r"\boversaw\b", r"\bmaintained\b",
        r"\badministered\b", r"\bmonitored\b", r"\bsupported\b", r"\bran\b",
        r"\borganized\b", r"\bscheduled\b",
    ],
    "Researcher": [
        r"\bpublished\b", r"\bresearched\b", r"\banalyzed\b", r"\binvestigated\b",
        r"\bexperimented\b", r"\bbenchmarked\b", r"\bstudied\b", r"\bmodeled\b",
        r"\bpatent(?:ed)?\b",
    ],
    "Teacher": [
        r"\bmentored\b", r"\btaught\b", r"\btrained\b", r"\bonboarded\b",
        r"\bcoached\b", r"\bdocumented\b", r"\bpresented\b",
        r"\bwrote\s+(?:docs|documentation|guides)\b",
    ],
}


def _heuristic_signal(text: str, links_summary: Optional[dict] = None) -> SignalProfile:
    text_lower = text.lower()
    archetypes = []

    for name, patterns in _ARCHETYPE_VERBS.items():
        hits = []
        for p in patterns:
            for m in re.finditer(p, text_lower):
                ctx = text_lower[max(0, m.start() - 40):m.end() + 40].strip()
                hits.append(ctx.replace("\n", " ")[:120])
        if not hits:
            continue
        score = min(1.0, len(hits) / 12.0)
        archetypes.append(SignalArchetype(name=name, score=round(score, 2),
                                          evidence=hits[:4]))

    archetypes.sort(key=lambda a: -a.score)

    notable = []
    if links_summary:
        gh = [l for l in links_summary["links"] if l["category"] == "github"]
        if gh and any(l.get("github_meta") for l in gh):
            meta = next(l["github_meta"] for l in gh if l.get("github_meta"))
            if meta:
                notable.append(
                    f"GitHub profile '{meta.get('username')}' has "
                    f"{meta.get('public_repos')} public repos, "
                    f"{meta.get('followers')} followers."
                )
        live_portfolios = [
            l for l in links_summary["links"]
            if l["category"] == "portfolio" and l["live"]
        ]
        if live_portfolios:
            notable.append(f"{len(live_portfolios)} live portfolio link(s) verified.")
        dead = [l for l in links_summary["links"] if l["live"] is False]
        if dead:
            notable.append(f"{len(dead)} link(s) returned errors or were unreachable.")

    metrics = re.findall(r"\b\d+(?:\.\d+)?[%xX]|\$\d+[KMB]?|\b\d{2,}[KMB]\b", text)
    if metrics:
        notable.append(f"{len(metrics)} quantified impact claims found.")

    if re.search(r"\b(led|managed|mentored)\s+(?:a\s+)?(?:team|group)\s+of\s+\d+", text_lower):
        notable.append("Explicit team-size leadership claim present.")

    gaps = []
    if not links_summary or not links_summary.get("links"):
        gaps.append("No external links (portfolio, GitHub, or writing) found.")
    if not metrics:
        gaps.append("No quantified impact claims -- bullets describe tasks, not outcomes.")
    if not re.search(r"\b(project|built|shipped|launched)\b", text_lower):
        gaps.append("No project or shipping verbs found.")

    if archetypes:
        top = ", ".join(a.name for a in archetypes[:2])
        summary = (
            f"Resume signals a primarily {top} profile "
            f"based on {sum(len(a.evidence) for a in archetypes)} observable action verbs."
        )
    else:
        summary = "Insufficient action verbs to characterise a profile archetype."

    confidence = "high" if len(text) > 2500 else "medium" if len(text) > 800 else "low"

    return SignalProfile(summary=summary, archetypes=archetypes,
                         notable=notable, gaps=gaps, confidence=confidence)


_LLM_PROMPT = """You are a resume analyst. Read the resume below and produce a JSON object with this exact schema:

{{
  "summary": "2-3 sentence description of what this profile signals. What does this person DO? Are they a builder, initiator, operator, researcher, scaler, teacher, or a mix? Be specific and evidence-based.",
  "archetypes": [
    {{"name": "Builder", "score": 0.85, "evidence": ["shipped X", "built Y"]}}
  ],
  "notable": ["standout observation 1", "standout observation 2"],
  "gaps": ["missing evidence 1", "missing evidence 2"]
}}

Rules:
- Base every claim on text actually present in the resume.
- Do NOT infer personality traits, emotional state, or protected characteristics.
- Do NOT speculate about AI authorship.
- Score archetypes from 0.0 to 1.0 based on evidence density.
- Return ONLY the JSON object. No markdown fences.

Resume:
---
{resume_text}
---
"""


def _llm_signal(text: str, api_key: str) -> SignalProfile:
    import json
    import httpx

    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": _LLM_PROMPT.format(resume_text=text[:8000])}
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Groq returned {r.status_code}: {r.text[:200]}")

    data = json.loads(r.json()["choices"][0]["message"]["content"])

    return SignalProfile(
        summary=data.get("summary", ""),
        archetypes=[
            SignalArchetype(
                name=a.get("name", "Unknown"),
                score=float(a.get("score", 0.0)),
                evidence=a.get("evidence", []),
            )
            for a in data.get("archetypes", [])
        ],
        notable=data.get("notable", []),
        gaps=data.get("gaps", []),
        confidence="high",
    )


def build_signal_profile(
    text: str,
    links_summary: Optional[dict] = None,
    groq_api_key: Optional[str] = None,
) -> dict:
    key = groq_api_key or os.environ.get("GROQ_API_KEY")
    used_llm = False

    if key:
        try:
            profile = _llm_signal(text, key)
            used_llm = True
        except Exception as e:
            profile = _heuristic_signal(text, links_summary)
            profile.gaps.append(f"LLM summarisation failed ({e}); used heuristic mode.")
    else:
        profile = _heuristic_signal(text, links_summary)

    return {
        "summary": profile.summary,
        "archetypes": [
            {"name": a.name, "score": a.score, "evidence": a.evidence}
            for a in profile.archetypes
        ],
        "notable": profile.notable,
        "gaps": profile.gaps,
        "confidence": profile.confidence,
        "mode": "llm" if used_llm else "heuristic",
        "disclaimer": (
            "Signal describes observable artefacts in the resume text. "
            "It does not assess personality, character, or suitability. "
            "Treat as a reading aid, not a hiring criterion."
        ),
    }