import asyncio
import uuid
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

from detectors_v3 import (
    analyze_single_resume_async,
    find_duplicates,
)

app = FastAPI(title="Signal API v3 -- Evidence & Verification")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BATCH_STORE: dict = {}


@app.get("/")
def root():
    return {"status": "Signal API v3 running"}


@app.post("/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(""),
):
    """Analyze a batch of resumes concurrently.

    Optional `jd_text` enables job-description matching for every resume.
    """
    # Read all files up-front (async), then process in parallel.
    raw = [(await f.read(), f.filename) for f in files]
    jd = jd_text.strip() or None

    results = list(await asyncio.gather(*[
        analyze_single_resume_async(data, name, jd) for data, name in raw
    ]))

    # Assign IDs before duplicate detection needs them.
    for r in results:
        r["id"] = str(uuid.uuid4())[:8]
        r["name"] = r["filename"].rsplit(".", 1)[0]

    pairs = find_duplicates([
        {"id": r["id"], "name": r["name"], "text": r["full_text"]}
        for r in results
    ])

    by: dict[str, list] = {}
    for p in pairs:
        by.setdefault(p["id_a"], []).append(p)
        by.setdefault(p["id_b"], []).append(p)

    for r in results:
        r["duplication"] = {
            "flagged": r["id"] in by,
            "matches": by.get(r["id"], []),
        }
        r["overall_flagged"] = (
            r["injection"]["flagged"]
            or r["duplication"]["flagged"]
            or r["consistency"]["flagged"]
            or r["authenticity"]["summary"]["discrepancy"] > 0
        )
        r.pop("full_text", None)

    BATCH_STORE["last"] = results
    return JSONResponse({
        "count": len(results),
        "resumes": results,
        "duplicate_pairs": pairs,
    })


@app.post("/verify-candidate")
async def verify_candidate(
    file: UploadFile = File(...),
    jd_text: str = Form(""),
):
    data = await file.read()
    jd = jd_text.strip() or None
    r = await analyze_single_resume_async(data, file.filename, jd)
    return JSONResponse({
        "filename": file.filename,
        "authenticity": r["authenticity"],
        "consistency": r["consistency"],
        "injection": r["injection"],
        "link_intel": r["link_intel"],
        "signal": r["signal"],
        "jd_match": r.get("jd_match"),
        "verification_questions": r["verification_questions"],
    })


@app.get("/last-batch")
def last_batch():
    return BATCH_STORE.get("last", [])