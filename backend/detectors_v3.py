"""Signal v3: evidence-first resume integrity + authenticity review.

Important: this module does NOT attempt to determine whether AI wrote a resume.
It distinguishes *corroborated claims* from *unverified/discrepant claims*.

New in this revision:
  - Automatic employer verification (ghost_company.py) -- no candidate input.
  - Link intelligence (link_intel.py) -- extracts portfolio/GitHub/blog links.
  - Signal profile summary (signal_profile.py) -- archetype read of the resume.
  - Job-description matching (jd_match.py) -- optional, keyless.
"""
from __future__ import annotations

import asyncio
import io
import re
from datetime import datetime
from typing import Any, Optional

import pdfplumber
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- new modules ---------------------------------------------------------
from ghost_company import verify_resume_employers, verify_resume_employers_async
from link_intel import analyze_resume_links
from signal_profile import build_signal_profile
from jd_match import match_jd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

CURRENT_YEAR = datetime.now().year
_EMBED_MODEL = None


# ------------------------------------------------------------------ helpers

def _near_white(c):
    if c is None:
        return False
    try:
        if isinstance(c, (int, float)):
            return c >= 0.90
        if isinstance(c, (list, tuple)):
            if len(c) == 3:
                return all(float(x) >= 0.90 for x in c)
            if len(c) == 4:
                return all(float(x) <= 0.10 for x in c)
    except Exception:
        pass
    return False


def _off_page(bbox, w, h):
    if not bbox or len(bbox) != 4:
        return False
    x0, top, x1, bottom = bbox
    return x1 < -2 or x0 > w + 2 or bottom < -2 or top > h + 2


INJECTION_PATTERNS = [
    r"\bignore\b.{0,50}\b(previous|prior|above|other)\b",
    r"\bdisregard\b.{0,50}\binstruction",
    r"\b(system|developer)\s+prompt\b",
    r"\byou are (an )?(ai|assistant|recruiter|hiring manager)\b",
    r"\b(rate|score)\s+(this\s+)?candidate\b",
    r"\b(recommend|advance|select|hire)\s+(this\s+)?candidate\b",
    r"\bautomatically\s+(approve|qualify|pass)\b",
    r"\boverride\b.{0,30}\b(ranking|scoring|evaluation)\b",
    r"\bdo not (flag|reject|penalize)\b",
    r"\bmark\s+(this|the)\s+(candidate|resume)\s+as\b",
    r"\btop[- ]?candidate\b",
    r"\bskip\s+(the\s+)?screen\b",
    r"\bpass\s+(this\s+)?resume\b",
]


def _manipulation(text):
    t = re.sub(r"\s+", " ", text.lower())
    return any(re.search(p, t, re.I) for p in INJECTION_PATTERNS)


def _ocr_pdf(data):
    if not (fitz and pytesseract and Image):
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pytesseract.image_to_string(img))
        return "\n".join(pages).strip()
    except Exception:
        return ""


def extract_visible_and_hidden_text(file_bytes: bytes, filename: str):
    hidden = []
    visible = []
    ocr_used = False

    if filename.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for pn, page in enumerate(pdf.pages, 1):
                    def _is_hidden_char(ch, page=page):
                        txt = ch.get("text", "")
                        if not txt.strip():
                            return False
                        size = float(ch.get("size", 0) or 0)
                        bbox = ch.get("bbox")
                        tiny = 0 < size <= 2.0
                        white = _near_white(ch.get("non_stroking_color"))
                        off = _off_page(bbox, float(page.width), float(page.height))
                        return tiny or white or off

                    for ch in page.chars:
                        txt = ch.get("text", "")
                        if not txt.strip():
                            continue
                        if _is_hidden_char(ch):
                            size = float(ch.get("size", 0) or 0)
                            bbox = ch.get("bbox")
                            tiny = 0 < size <= 2.0
                            white = _near_white(ch.get("non_stroking_color"))
                            off = _off_page(bbox, float(page.width), float(page.height))
                            reasons = []
                            if tiny:
                                reasons.append(f"very small font ({size:.1f}pt)")
                            if white:
                                reasons.append("near-white text")
                            if off:
                                reasons.append("text outside page bounds")
                            hidden.append({
                                "text": txt,
                                "reason": ", ".join(reasons),
                                "page": pn,
                                "bbox": list(bbox) if bbox else None,
                                "manipulation_language": _manipulation(txt),
                            })

                    # Reconstruct properly spaced/line-broken text for everything
                    # that ISN'T hidden, using pdfplumber's own layout engine.
                    # (Joining page.chars with " " destroys word boundaries,
                    # since chars are per-glyph, not per-word.)
                    visible_page = page.filter(
                        lambda obj: obj.get("object_type") != "char" or not _is_hidden_char(obj)
                    )
                    page_text = visible_page.extract_text() or ""
                    if page_text:
                        visible.append(page_text)
        except Exception as e:
            return {
                "visible": "",
                "hidden_candidates": [],
                "all_text": "",
                "ocr_used": False,
                "extraction_error": str(e),
            }
        vis = "\n".join(visible)
        if len(re.sub(r"\s+", "", vis)) < 80:
            ocr = _ocr_pdf(file_bytes)
            if len(ocr) > len(vis):
                vis = ocr
                ocr_used = True

    elif filename.lower().endswith(".docx"):
        doc = DocxDocument(io.BytesIO(file_bytes))
        for p in doc.paragraphs:
            for run in p.runs:
                txt = run.text
                if not txt.strip():
                    continue
                hidden_flag = bool(run.font.hidden)
                size = run.font.size.pt if run.font.size else None
                rgb = None
                try:
                    rgb = str(run.font.color.rgb) if run.font.color and run.font.color.rgb else None
                except Exception:
                    pass
                tiny = size is not None and size <= 2
                white = rgb in {"FFFFFF", "FEFEFE", "FFFFFE"}
                if hidden_flag or tiny or white:
                    rs = []
                    if hidden_flag:
                        rs.append("marked as hidden text")
                    if tiny:
                        rs.append(f"very small font ({size:.1f}pt)")
                    if white:
                        rs.append("near-white font")
                    hidden.append({
                        "text": txt,
                        "reason": ", ".join(rs),
                        "page": None,
                        "bbox": None,
                        "manipulation_language": _manipulation(txt),
                    })
                else:
                    visible.append(txt)
        vis = "\n".join(visible)
    else:
        vis = file_bytes.decode("utf-8", errors="ignore")

    return {
        "visible": vis,
        "hidden_candidates": hidden,
        "all_text": vis + " " + " ".join(x["text"] for x in hidden),
        "ocr_used": ocr_used,
        "extraction_error": None,
    }


def scan_for_injection(hidden_candidates):
    matches = []
    for h in hidden_candidates:
        txt = re.sub(r"\s+", " ", h["text"]).strip()
        if not txt:
            continue
        if _manipulation(txt) or len(txt) >= 25:
            x = dict(h)
            x["text"] = txt
            x["confidence"] = "high" if _manipulation(txt) else "review"
            matches.append(x)
    return {
        "flagged": bool(matches),
        "matches": matches,
        "hidden_text_blocks": len(hidden_candidates),
        "high_confidence": sum(x["confidence"] == "high" for x in matches),
    }


# ------------------------------------------------------------------ duplication

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s+#./-]", " ", s.lower())).strip()


def _chunks(text):
    lines = [_norm(x) for x in text.splitlines() if _norm(x)]
    if len(lines) >= 3:
        pieces = lines
    else:
        pieces = re.split(r"(?<=[.!?])\s+", _norm(text))
    return [p for p in pieces if len(p.split()) >= 12][:120]


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None and SentenceTransformer:
        try:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBED_MODEL = False
    return _EMBED_MODEL if _EMBED_MODEL is not False else None


def _chunk_hits(a, b):
    if not a or not b:
        return []
    model = _get_embed_model()
    if model:
        try:
            allc = a + b
            e = model.encode(allc, normalize_embeddings=True, show_progress_bar=False)
            sim = e[: len(a)] @ e[len(a):].T
        except Exception:
            model = None
    if not model:
        allc = a + b
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        m = v.fit_transform(allc)
        sim = cosine_similarity(m[: len(a)], m[len(a):])
    hits = []
    for i in range(len(a)):
        for j in range(len(b)):
            s = float(sim[i][j])
            if s >= 0.82:
                hits.append({"chunk_a": a[i], "chunk_b": b[j], "similarity": round(s, 3)})
    return sorted(hits, key=lambda x: -x["similarity"])[:6]


def _candidate_identity(text: str) -> str:
    """Best-effort identity for a resume (email, else first non-blank line),
    used to tell 'the same candidate reappearing' apart from 'two
    different candidates sharing suspiciously similar content'."""
    m = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
    if m:
        return m.group(0).lower()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return _norm(line)
    return ""


def find_duplicates(resumes, threshold=0.82, common_chunk_max_docs=None):
    """Flag resumes that share distinctive text, not generic boilerplate.

    Many resumes independently use similar generic phrasing ("built
    production services", "designed APIs and automated tests", etc.),
    especially when produced from templates or AI writing tools. A chunk
    that recurs across more than `common_chunk_max_docs` resumes in this
    batch is treated as boilerplate and excluded from duplication
    evidence -- only text shared by a minority of the batch counts as a
    real signal of copy/reuse. Defaults to roughly a quarter of the
    batch (floor of 2): a phrase used across most of the batch is almost
    certainly a shared template/convention, while a small cluster of
    resumes sharing unusually specific text is a genuine duplicate ring.

    Matches between two documents that share the same candidate identity
    (same email, or same declared name) are never flagged -- that's the
    same person's resume reappearing, not two candidates copying content.
    """
    if common_chunk_max_docs is None:
        common_chunk_max_docs = max(2, len(resumes) // 4)
    all_chunks = {r["id"]: _chunks(r["text"]) for r in resumes}
    identity = {r["id"]: _candidate_identity(r["text"]) for r in resumes}

    # Bucket by a stable prefix, not the full chunk: PDF line-wrapping cuts
    # the same boilerplate sentence off at different points in different
    # documents (varying name/company lengths shift the wrap), so exact
    # full-chunk matching badly undercounts how common a phrase really is.
    def _key(c, n=10):
        return " ".join(c.split()[:n])

    doc_count: dict[str, int] = {}
    for chunks in all_chunks.values():
        for key in {_key(c) for c in chunks}:
            doc_count[key] = doc_count.get(key, 0) + 1
    common_keys = {k for k, n in doc_count.items() if n > common_chunk_max_docs}

    matches = []
    for i in range(len(resumes)):
        for j in range(i + 1, len(resumes)):
            id_a, id_b = resumes[i]["id"], resumes[j]["id"]
            if identity[id_a] and identity[id_a] == identity[id_b]:
                continue
            ca = [c for c in all_chunks[id_a] if _key(c) not in common_keys]
            cb = [c for c in all_chunks[id_b] if _key(c) not in common_keys]
            hits = _chunk_hits(ca, cb)
            strong = [h for h in hits if h["similarity"] >= threshold]
            near = [h for h in hits if h["similarity"] >= 0.94]
            if len(strong) >= 2 or near:
                matches.append({
                    "id_a": id_a,
                    "name_a": resumes[i]["name"],
                    "id_b": id_b,
                    "name_b": resumes[j]["name"],
                    "similarity": round(max(h["similarity"] for h in strong), 3),
                    "matching_chunks": strong[:3],
                    "evidence_type": "distinctive section reuse",
                })
    return sorted(matches, key=lambda x: -x["similarity"])



# ------------------------------------------------------------------ consistency

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
)}
MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_RANGE = re.compile(
    rf"\b({MONTH})?\s*[,./-]?\s*((?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"({MONTH})?\s*((?:19|20)\d{{2}}|present|current)\b",
    re.I,
)
GRAD = [
    re.compile(
        r"(?:graduat(?:ed|ion)|bachelor(?:\'s)?|master(?:\'s)?|b\.?s\.?|m\.?s\.?|degree)"
        r".{0,80}\b((?:19|20)\d{2})\b",
        re.I,
    ),
    re.compile(
        r"\b((?:19|20)\d{2})\b.{0,30}(?:graduat(?:ed|ion)|bachelor|master|degree)\b",
        re.I,
    ),
]
EXP = re.compile(r"\b(\d{1,2})\+?\s+years?\s+(?:of\s+)?experience\b", re.I)
WORK = re.compile(
    r"(engineer|developer|manager|analyst|scientist|designer|consultant|director|"
    r"intern|lead|architect|specialist|coordinator|researcher|software|product)",
    re.I,
)


def _month(s, default):
    if not s:
        return default
    s = s.lower()
    for k, v in MONTHS.items():
        if s.startswith(k):
            return v
    return default


def _ranges(text):
    out = []
    for m in DATE_RANGE.finditer(text):
        sy = int(m.group(2))
        ey = CURRENT_YEAR if m.group(4).lower() in ("present", "current") else int(m.group(4))
        st = sy * 12 + _month(m.group(1), 1)
        en = ey * 12 + _month(m.group(3), 12)
        ctx = text[max(0, m.start() - 100): min(len(text), m.end() + 100)]
        out.append({
            "raw": m.group(0),
            "start": st,
            "end": en,
            "context": ctx,
            "work_like": bool(WORK.search(ctx)),
        })
    return out


def check_consistency(text):
    flags = []
    grad = None
    for p in GRAD:
        m = p.search(text)
        if m:
            grad = int(m.group(1))
            break
    em = EXP.search(text)
    claimed = int(em.group(1)) if em else None
    if grad and claimed and claimed > (CURRENT_YEAR - grad) + 2:
        flags.append({
            "type": "graduation_vs_experience",
            "severity": "high",
            "detail": f"Claims {claimed} years of experience but lists graduation in {grad}.",
            "evidence": {"graduation_year": grad, "claimed_years": claimed},
        })
    rs = _ranges(text)
    now = CURRENT_YEAR * 12 + datetime.now().month
    for r in rs:
        if r["start"] > now:
            flags.append({
                "type": "future_dated_role",
                "severity": "high",
                "detail": f"Role/date range begins in the future: {r['raw']}.",
                "evidence": r["raw"],
            })
        if r["end"] < r["start"]:
            flags.append({
                "type": "reversed_date_range",
                "severity": "high",
                "detail": f"End date precedes start date: {r['raw']}.",
                "evidence": r["raw"],
            })
    wr = [r for r in rs if r["work_like"]]
    for i in range(len(wr)):
        for j in range(i + 1, len(wr)):
            overlap = min(wr[i]["end"], wr[j]["end"]) - max(wr[i]["start"], wr[j]["start"]) + 1
            if overlap >= 12:
                flags.append({
                    "type": "overlapping_roles",
                    "severity": "medium",
                    "detail": (
                        f"Two work-history ranges overlap for about {overlap} months: "
                        f"'{wr[i]['raw']}' and '{wr[j]['raw']}'."
                    ),
                    "evidence": [wr[i]["raw"], wr[j]["raw"]],
                })
    return {
        "flagged": bool(flags),
        "flags": flags,
        "extracted": {
            "graduation_year": grad,
            "claimed_years_experience": claimed,
            "date_ranges_found": [r["raw"] for r in rs],
        },
    }


# ------------------------------------------------------------------ identity / quality

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)
PHONE_RE = re.compile(r"(?:\+?1[ .-]?)?(\(?\d{3}\)?)[ .-]?\d{3}[ .-]?\d{4}")
LOCATION_RE = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*([A-Z]{2})\b")
COMPANY_RE = re.compile(
    r"(?:\bat|@|\|)\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3})"
    r"(?=\s*(?:\||\n|,|\.|$))"
)

GENERIC_PATTERNS = [
    r"\b(drove|leveraged|spearheaded|orchestrated|optimized|transformed|revolutionized)\b"
    r".{0,120}\b\d+%\b",
    r"\b(advanced|cutting-edge|innovative|strategic|robust|dynamic|scalable|synergy|"
    r"best-in-class)\b.{0,100}\b(impact|solution|platform|strategy)\b",
]


def extract_identity_claims(text):
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    loc = LOCATION_RE.search(text)
    companies = []
    for m in COMPANY_RE.finditer(text):
        c = m.group(1).strip(" |,-")
        if len(c) >= 3 and c.lower() not in {
            "new york", "san francisco", "los angeles", "software engineering"
        }:
            companies.append(c)
    companies = list(dict.fromkeys(companies))[:12]
    return {
        "email_domain": email.group(1).lower() if email else None,
        "phone_area_code": re.sub(r"\D", "", phone.group(1)) if phone else None,
        "location": f"{loc.group(1)}, {loc.group(2)}" if loc else None,
        "employers": companies,
    }


def claim_quality(text):
    hits = []
    for p in GENERIC_PATTERNS:
        for m in re.finditer(p, text, re.I):
            hits.append({"type": "generic_or_vague_claim", "evidence": m.group(0)[:220]})
    return {
        "review": len(hits) >= 2,
        "flags": hits[:5],
        "principle": "content vagueness is a review cue, not proof of fabrication",
    }


# ------------------------------------------------------------------ verification pack

async def build_verification_pack_async(text: str) -> dict:
    """Automatic employer verification from resume text. No candidate input."""
    claims = extract_identity_claims(text)
    employers_block = await verify_resume_employers_async(text)

    checks = list(employers_block["checks"])
    summary = {
        "corroborated": sum(c["status"] == "corroborated" for c in checks),
        "discrepancy": sum(c["status"] == "discrepancy" for c in checks),
        "unverified": sum(c["status"] == "unverified" for c in checks),
    }
    return {
        "claims": claims,
        "checks": checks,
        "claim_quality": claim_quality(text),
        "summary": summary,
        "principle": employers_block["principle"],
    }


def build_verification_pack(text: str) -> dict:
    """Sync wrapper -- keeps existing callers working."""
    claims = extract_identity_claims(text)
    employers_block = verify_resume_employers(text)
    checks = list(employers_block["checks"])
    summary = {
        "corroborated": sum(c["status"] == "corroborated" for c in checks),
        "discrepancy": sum(c["status"] == "discrepancy" for c in checks),
        "unverified": sum(c["status"] == "unverified" for c in checks),
    }
    return {
        "claims": claims,
        "checks": checks,
        "claim_quality": claim_quality(text),
        "summary": summary,
        "principle": employers_block["principle"],
    }


# ------------------------------------------------------------------ follow-up questions

def generate_verification_questions(text, max_questions=3):
    qs = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = sent.strip(" -•\t")
        if len(s.split()) < 7:
            continue
        if re.search(
            r"\b(API|Kafka|AWS|Azure|GCP|Python|Java|SQL|Kubernetes|React|TensorFlow|"
            r"PyTorch|model|pipeline|migration|latency|revenue|users|%)\b",
            s,
            re.I,
        ):
            qs.append(
                f"You mention: '{s[:180]}'. What was your specific contribution, "
                "and what was one technical decision you personally made?"
            )
        if len(qs) >= max_questions:
            break
    if not qs:
        qs.append(
            "Pick one project on your resume: what was the hardest problem, "
            "what did you personally change, and how did you verify the result?"
        )
    return qs


# ------------------------------------------------------------------ top-level pipeline

async def analyze_single_resume_async(
    file_bytes: bytes,
    filename: str,
    jd_text: Optional[str] = None,
) -> dict:
    x = extract_visible_and_hidden_text(file_bytes, filename)
    text = x["visible"]

    # Run independent pipelines concurrently.
    link_intel, authenticity = await asyncio.gather(
        asyncio.to_thread(analyze_resume_links, text),
        build_verification_pack_async(text),
    )

    signal = await asyncio.to_thread(build_signal_profile, text, link_intel)

    result: dict[str, Any] = {
        "filename": filename,
        "visible_text_preview": text[:700],
        "ocr_used": x["ocr_used"],
        "injection": scan_for_injection(x["hidden_candidates"]),
        "consistency": check_consistency(text),
        "authenticity": authenticity,
        "verification_questions": generate_verification_questions(text),
        "link_intel": link_intel,
        "signal": signal,
        "full_text": text,
    }

    if jd_text and len(jd_text.strip()) >= 40:
        result["jd_match"] = await asyncio.to_thread(match_jd, text, jd_text)

    return result


def analyze_single_resume(file_bytes: bytes, filename: str, jd_text: Optional[str] = None) -> dict:
    """Sync entry point -- used by callers not on an event loop."""
    try:
        return asyncio.run(analyze_single_resume_async(file_bytes, filename, jd_text))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                analyze_single_resume_async(file_bytes, filename, jd_text)
            )
        finally:
            loop.close()