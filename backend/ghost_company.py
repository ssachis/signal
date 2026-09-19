"""
ghost_company.py -- Automatic employer verification from resume text alone.
Uses only free, keyless APIs:
  - GLEIF (api.gleif.org)          -- global legal-entity registry
  - RDAP (rdap.org)                -- domain registration dates
  - Wayback CDX (web.archive.org)  -- historical web presence
  - DNS MX                         -- mail-server liveness
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Optional

import dns.resolver
import httpx
from rapidfuzz import fuzz

UA = {"User-Agent": "Signal-Verification/1.0 (trust@example.com)"}


@dataclass
class EmploymentRecord:
    company: str
    title: Optional[str]
    start: Optional[dt.date]
    end: Optional[dt.date]
    raw_context: str = ""


@dataclass
class Evidence:
    source: str
    url: Optional[str]
    detail: str
    weight: float


@dataclass
class EmployerVerdict:
    record: EmploymentRecord
    status: str
    score: float
    evidence: list = field(default_factory=list)
    flags: list = field(default_factory=list)


_LEGAL_SUFFIX = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|gmbh|ag|"
    r"bv|nv|sarl|sa|srl|spa|pty|plc|pvt|private|llp|lp|oy|ab|as|aps|kk|pte|"
    r"sdn|bhd|group|holdings|technologies|solutions|systems|labs|studio)\b\.?",
    re.I,
)


def normalize(name: str) -> str:
    n = _LEGAL_SUFFIX.sub(" ", name.lower())
    n = re.sub(r"[^\w\s]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(name))


class _Cache:
    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._data: dict = {}

    def get(self, key: str):
        item = self._data.get(key)
        if not item:
            return None
        ts, val = item
        if (dt.datetime.now().timestamp() - ts) > self._ttl:
            del self._data[key]
            return None
        return val

    def set(self, key: str, val) -> None:
        self._data[key] = (dt.datetime.now().timestamp(), val)


_cache = _Cache()


async def gleif(client: httpx.AsyncClient, name: str) -> list:
    try:
        r = await client.get(
            "https://api.gleif.org/api/v1/lei-records",
            params={"filter[entity.legalName]": name, "page[size]": 5},
            headers=UA,
            timeout=8,
        )
        if r.status_code != 200:
            return []
        target = normalize(name)
        out = []
        for rec in r.json().get("data", []):
            attrs = rec["attributes"]
            legal = attrs["entity"]["legalName"]["name"]
            status = attrs["entity"].get("status", "UNKNOWN")
            if fuzz.token_set_ratio(normalize(legal), target) < 88:
                continue
            lei = attrs["lei"]
            weight = 0.40 if status == "ACTIVE" else 0.20
            out.append(Evidence(
                source="GLEIF",
                url=f"https://search.gleif.org/#/record/{lei}",
                detail=f"Legal entity: {legal} (status={status}, LEI {lei})",
                weight=weight,
            ))
        return out
    except Exception:
        return []


async def rdap_registration(client: httpx.AsyncClient, domain: str) -> Optional[dt.date]:
    try:
        r = await client.get(f"https://rdap.org/domain/{domain}", headers=UA, timeout=6)
        if r.status_code != 200:
            return None
        for ev in r.json().get("events", []):
            if ev.get("eventAction") == "registration":
                return dt.date.fromisoformat(ev["eventDate"][:10])
    except Exception:
        pass
    return None


async def wayback_first(client: httpx.AsyncClient, domain: str) -> Optional[dt.date]:
    try:
        r = await client.get(
            "http://web.archive.org/cdx/search/cdx",
            params={"url": f"{domain}/*", "output": "json", "fl": "timestamp",
                    "sort": "asc", "limit": 1},
            headers=UA,
            timeout=8,
        )
        rows = r.json()
        if len(rows) > 1:
            return dt.datetime.strptime(rows[1][0], "%Y%m%d%H%M%S").date()
    except Exception:
        pass
    return None


def has_mx(domain: str) -> bool:
    try:
        return len(dns.resolver.resolve(domain, "MX", lifetime=4)) > 0
    except Exception:
        return False


_DOMAIN_HINT = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]{1,60}\.(?:com|io|co|ai|net|org|dev|app))",
    re.I,
)

_IGNORED_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "linkedin.com", "github.com", "google.com",
}


def infer_domain(name: str, context: str = "") -> Optional[str]:
    for m in _DOMAIN_HINT.finditer(context):
        candidate = m.group(1).lower()
        if candidate not in _IGNORED_DOMAINS:
            return candidate
    slug = slugify(name)
    if not slug or len(slug) < 3:
        return None
    for tld in (".com", ".io", ".co", ".ai"):
        if has_mx(slug + tld):
            return slug + tld
    return None


async def verify_employer(record: EmploymentRecord, client: httpx.AsyncClient) -> EmployerVerdict:
    cache_key = normalize(record.company)
    cached = _cache.get(cache_key)
    if cached:
        return EmployerVerdict(
            record=record, status=cached.status, score=cached.score,
            evidence=cached.evidence, flags=cached.flags,
        )

    domain = infer_domain(record.company, record.raw_context)

    registry_hits, rdap_date, wayback_date, mx_ok = await asyncio.gather(
        gleif(client, record.company),
        rdap_registration(client, domain) if domain else asyncio.sleep(0),
        wayback_first(client, domain) if domain else asyncio.sleep(0),
        asyncio.to_thread(has_mx, domain) if domain else asyncio.sleep(0),
        return_exceptions=True,
    )

    if not isinstance(registry_hits, list):
        registry_hits = []
    rdap_date = rdap_date if isinstance(rdap_date, dt.date) else None
    wayback_date = wayback_date if isinstance(wayback_date, dt.date) else None
    mx_ok = mx_ok is True

    verdict = _score(record, registry_hits, domain, rdap_date, wayback_date, mx_ok)
    _cache.set(cache_key, verdict)
    return verdict


def _score(record, registry_hits, domain, rdap_date, wayback_date, mx_ok):
    ev = list(registry_hits)
    flags = []

    if not registry_hits:
        flags.append("no_registry_hit")
    else:
        ev.append(Evidence("registry", None,
                           "Legal entity corroborated in a public registry", 0.0))

    if domain:
        if mx_ok:
            ev.append(Evidence("DNS", None, f"{domain} has mail (MX) records", 0.10))
        else:
            ev.append(Evidence("DNS", None, f"{domain} has no MX records", -0.15))
            flags.append("no_mail")

        if rdap_date and record.start:
            if rdap_date > record.start:
                delta = (rdap_date - record.start).days
                ev.append(Evidence(
                    "RDAP", f"https://rdap.org/domain/{domain}",
                    f"Domain registered {rdap_date.isoformat()} -- "
                    f"{delta} days AFTER claimed start {record.start.isoformat()}",
                    -0.40,
                ))
                flags.append("domain_postdates_employment")
            else:
                ev.append(Evidence(
                    "RDAP", f"https://rdap.org/domain/{domain}",
                    f"Domain registered {rdap_date.isoformat()} (predates claim)",
                    0.15,
                ))

        if wayback_date and record.start:
            if wayback_date > record.start + dt.timedelta(days=180):
                ev.append(Evidence(
                    "Wayback", f"http://web.archive.org/web/*/{domain}",
                    f"No archived web presence until {wayback_date.isoformat()} "
                    f"(claimed start {record.start.isoformat()})",
                    -0.20,
                ))
                flags.append("no_contemporaneous_web_presence")
            else:
                ev.append(Evidence(
                    "Wayback", f"http://web.archive.org/web/*/{domain}",
                    f"Archived since {wayback_date.isoformat()}", 0.15,
                ))
        elif domain:
            ev.append(Evidence("Wayback", None,
                               "Never archived by the Wayback Machine", -0.10))
            flags.append("never_archived")
    else:
        flags.append("no_domain_found")

    raw = sum(e.weight for e in ev)
    score = max(0.0, min(1.0, raw + 0.40))

    if score >= 0.70:
        status = "corroborated"
    elif score >= 0.40:
        status = "unverified"
    else:
        status = "discrepancy"

    return EmployerVerdict(record=record, status=status,
                           score=round(score, 3), evidence=ev, flags=flags)


async def verify_employers(records, concurrency: int = 6):
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        async def _one(rec):
            async with sem:
                try:
                    return await asyncio.wait_for(verify_employer(rec, client), timeout=20.0)
                except Exception as e:
                    return EmployerVerdict(
                        record=rec, status="unverified", score=0.0,
                        evidence=[Evidence("pipeline", None,
                                           f"Verification error: {e}", 0.0)],
                        flags=["pipeline_error"],
                    )
        return await asyncio.gather(*[_one(r) for r in records])


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_MONTH_PAT = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)

_DATE_RANGE = re.compile(
    rf"\b({_MONTH_PAT})?\s*[,./-]?\s*((?:19|20)\d{{2}})\s*"
    rf"(?:-|–|—|to)\s*"
    rf"({_MONTH_PAT})?\s*((?:19|20)\d{{2}}|present|current)\b",
    re.I,
)

_TITLE_AT = re.compile(
    r"^([A-Z][A-Za-z/&,\- ]{2,60}?)\s+(?:at|@)\s+"
    r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})",
    re.M,
)

_COMPANY_TITLE = re.compile(
    r"^([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})"
    r"\s*[—|·\-]\s*"
    r"([A-Z][A-Za-z/&,\- ]{2,60})$",
    re.M,
)

_NON_COMPANY = {
    "university", "college", "school", "institute", "linkedin", "github",
    "san francisco", "new york", "los angeles", "seattle", "boston",
    "present", "current", "experience", "education", "skills", "projects",
    "bachelor", "master", "doctor", "phd", "united states", "california",
}


def _looks_like_company(name: str) -> bool:
    if not name or len(name) < 3:
        return False
    if name.lower() in _NON_COMPANY:
        return False
    if not any(c.isupper() for c in name):
        return False
    if re.match(r"^(19|20)\d{2}$", name):
        return False
    return True


def _month_of(token, default):
    if not token:
        return default
    t = token.lower()
    for k, v in _MONTHS.items():
        if t.startswith(k):
            return v
    return default


def _parse_range(m):
    try:
        sy = int(m.group(2))
        sm = _month_of(m.group(1), 1)
        start = dt.date(sy, sm, 1)
        ey_raw = m.group(4).lower()
        if ey_raw in ("present", "current"):
            end = None
        else:
            ey = int(ey_raw)
            em = _month_of(m.group(3), 12)
            end = dt.date(ey, em, 1)
        return start, end
    except Exception:
        return None, None


def _company_and_title(ctx):
    for m in _TITLE_AT.finditer(ctx):
        title = m.group(1).strip()
        company = m.group(2).strip(" .,-|")
        if _looks_like_company(company):
            return company, title

    for m in _COMPANY_TITLE.finditer(ctx):
        a, b = m.group(1).strip(), m.group(2).strip()
        if _looks_like_company(a):
            return a, b
        if _looks_like_company(b):
            return b, a

    for line in ctx.splitlines()[::-1]:
        line = line.strip()
        if not line or _DATE_RANGE.search(line):
            continue
        m = re.match(r"^([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})", line)
        if m and _looks_like_company(m.group(1)):
            return m.group(1).strip(" .,-"), None

    return None, None


def extract_employment_records(text: str) -> list:
    records = []
    seen = set()
    for m in _DATE_RANGE.finditer(text):
        ctx = text[max(0, m.start() - 220):min(len(text), m.end() + 60)]
        start, end = _parse_range(m)
        company, title = _company_and_title(ctx)
        if not company:
            continue
        key = normalize(company)
        if key in seen:
            continue
        seen.add(key)
        records.append(EmploymentRecord(
            company=company, title=title, start=start, end=end,
            raw_context=ctx,
        ))
    return records


def _verdict_to_check(v: EmployerVerdict) -> dict:
    dates = ""
    if v.record.start:
        dates = v.record.start.isoformat()
        dates += " -- " + (v.record.end.isoformat() if v.record.end else "present")

    return {
        "check": f"Employer: {v.record.company}" + (f" ({dates})" if dates else ""),
        "category": "employer",
        "status": v.status,
        "score": v.score,
        "flags": v.flags,
        "detail": _human_detail(v),
        "evidence": [
            {"source": e.source, "url": e.url, "detail": e.detail, "weight": e.weight}
            for e in v.evidence
        ],
    }


def _human_detail(v: EmployerVerdict) -> str:
    if v.status == "corroborated":
        return "Independent public evidence corroborates this employer."
    if v.status == "discrepancy":
        reasons = ", ".join(v.flags) or "multiple signals"
        return (
            f"Public evidence contradicts this employment claim ({reasons}). "
            "Route to human review -- this is a signal, not a verdict."
        )
    return (
        "Insufficient public evidence to corroborate or contradict this employer. "
        "Common for small, private, or recently-renamed companies."
    )


async def verify_resume_employers_async(text: str) -> dict:
    records = extract_employment_records(text)
    if not records:
        return {
            "checks": [],
            "summary": {"corroborated": 0, "unverified": 0, "discrepancy": 0},
            "principle": "No employer records extracted from the resume text.",
        }

    verdicts = await verify_employers(records)
    checks = [_verdict_to_check(v) for v in verdicts]

    return {
        "checks": checks,
        "summary": {
            "corroborated": sum(c["status"] == "corroborated" for c in checks),
            "unverified": sum(c["status"] == "unverified" for c in checks),
            "discrepancy": sum(c["status"] == "discrepancy" for c in checks),
        },
        "principle": (
            "Employers are checked against GLEIF (global registry), RDAP "
            "(domain registration), the Wayback Machine (web history), and "
            "DNS (mail liveness). 'Unverified' means evidence was insufficient, "
            "not that the claim is false."
        ),
    }


def verify_resume_employers(text: str) -> dict:
    try:
        return asyncio.run(verify_resume_employers_async(text))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(verify_resume_employers_async(text))
        finally:
            loop.close() 