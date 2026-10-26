"""Run Signal on the demo batch and save the results the page shows in demo mode.

Same steps as POST /analyze-batch in backend/main.py, without the web server.
Run from the repo root:  python demo/run_demo.py
"""
import asyncio, json, os, sys, uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from detectors_v3 import analyze_single_resume_async, find_duplicates  # noqa: E402

BATCH = [
    'A_clean_real.docx', 'A_clean_03_Noah_Williams.pdf', 'A_clean_04_Mia_Patel.pdf',
    'B_ai_polished_legit.docx', 'A_clean_05_Ethan_Brooks.pdf',
    'D_injected.docx',
    'E1_duplicated.docx', 'E2_duplicated.docx',
    'E_duplicate_02_1_Jordan_Blake.pdf', 'E_duplicate_02_2_Jordan_Reed.pdf',
]
SRC = os.path.join(os.path.dirname(__file__), '..', 'sample_resumes')
OUT = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'demo-results.json')


async def main():
    if '--all' in sys.argv:  # check every labeled sample, for choosing the demo batch
        BATCH[:] = sorted(f for f in os.listdir(SRC) if f[:2] in ('A_','B_','C_','D_','E_','F_') or f[:2] in ('E1','E2'))
    raw = [(open(os.path.join(SRC, n), 'rb').read(), n) for n in BATCH]
    results = list(await asyncio.gather(*[analyze_single_resume_async(d, n, None) for d, n in raw]))
    for r in results:
        r['id'] = str(uuid.uuid4())[:8]
        r['name'] = r['filename'].rsplit('.', 1)[0]
    pairs = find_duplicates([{'id': r['id'], 'name': r['name'], 'text': r['full_text']} for r in results])
    by = {}
    for p in pairs:
        by.setdefault(p['id_a'], []).append(p)
        by.setdefault(p['id_b'], []).append(p)
    for r in results:
        r['duplication'] = {'flagged': r['id'] in by, 'matches': by.get(r['id'], [])}
        r['overall_flagged'] = (r['injection']['flagged'] or r['duplication']['flagged']
                                or r['consistency']['flagged'] or r['authenticity']['summary']['discrepancy'] > 0)
        r.pop('full_text', None)
    data = {'count': len(results), 'resumes': results, 'duplicate_pairs': pairs, 'demo': True,
            'files': BATCH}
    if '--all' in sys.argv:
        OUT_ = os.path.join(os.path.dirname(__file__), 'all-samples.txt')
    else:
        OUT_ = OUT
    with open(OUT_ if OUT_ == OUT else os.devnull, 'w') as f:
        json.dump(data, f, indent=1, default=str)
    for r in results:
        print(f"{r['filename']:48s} flagged={r['overall_flagged']} inj={r['injection']['flagged']} "
              f"dup={r['duplication']['flagged']} con={r['consistency']['flagged']} "
              f"employer_discrepancy={r['authenticity']['summary']['discrepancy']}")

asyncio.run(main())
