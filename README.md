# Signal — Quick Start

## What's here
- `backend/detectors.py` — the 3 detection engines (injection, duplication, consistency)
- `backend/main.py` — FastAPI app exposing `/analyze-batch`
- `frontend/index.html` — standalone UI, mocked ATS candidate list (no build step, just open it)
- `sample_resumes/` — 6 pre-generated test cases covering every category from the MVP spec
- `generate_samples.py` — regenerate/edit the sample resumes

## Run it (2 terminals)

**Terminal 1 — backend:**
```
cd backend
pip install fastapi uvicorn pdfplumber python-docx scikit-learn python-multipart --break-system-packages
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend:**
Just open `frontend/index.html` directly in a browser (double-click it, or `open frontend/index.html`).
It talks to `http://localhost:8000` — make sure the backend is running first.

## Demo flow
1. In the browser, select all 6 files from `sample_resumes/` at once
2. Click "Analyze Batch"
3. Click each row to expand evidence:
   - A, B → clean (B is AI-polished but legit — the proof point)
   - D → injection flagged, hidden text shown highlighted
   - E1/E2 → duplication flagged, similarity score shown
   - F → consistency flagged, timeline contradiction shown

## Notes / things to tune if you have time
- Duplication threshold is 0.75 cosine similarity (TF-IDF) — tune in `main.py` (`find_duplicates(dup_input, threshold=0.75)`) if you get false positives/negatives on real data
- Consistency checker uses regex extraction — works well on clearly-stated years/dates, may miss oddly-phrased ones. If you have time, swap the regex extraction for a single structured LLM call (prompt: "extract graduation year, and each role's start/end date and title, as JSON") for more robustness — the downstream logic checks stay identical
- Injection patterns list in `detectors.py` (`INJECTION_PATTERNS`) — add more phrases if your test injections use different wording
- Used TF-IDF instead of sentence-transformer embeddings for duplication (lighter, no torch dependency, fast to install) — swap in embeddings later if you want semantic (not just lexical) duplicate matching
# signal
