"""Link intelligence: extract, categorise, verify, and enrich links from resumes.
All free. No API keys required."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

UA = {"User-Agent": "Signal-LinkIntel/1.0 (trust@example.com)"}


@dataclass
class LinkRecord:
    url: str
    category: str
    label: str
    live: Optional[bool] = None
    status_code: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    redirect_target: Optional[str] = None
    error: Optional[str] = None
    tech_stack: list = field(default_factory=list)
    github_meta: Optional[dict] = None


_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"([a-z0-9][a-z0-9\-]{1,60}\.(?:com|io|co|ai|net|org|dev|app|me|xyz|tech|so|gg|sh|link|ly)"
    r"(?:/[^\s\)\]\},;\"']*)?)",
    re.I,
)

_GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9\-_]+)(?:/([A-Za-z0-9\-_.]+))?", re.I)

_LABEL_HINTS = re.compile(
    r"(portfolio|website|blog|github|gitlab|linkedin|dribbble|behance|"
    r"medium|substack|personal site|project|demo|app|deployed|live at)",
    re.I,
)

_IGNORED_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "google.com", "facebook.com", "twitter.com", "x.com",
}


def _classify(url: str) -> str:
    host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
    if "github.com" in host or "gitlab.com" in host:
        return "github"
    if "linkedin.com" in host:
        return "linkedin"
    if "dribbble.com" in host or "behance.net" in host:
        return "portfolio"
    if "medium.com" in host or "substack.com" in host:
        return "writing"
    if host in _IGNORED_DOMAINS:
        return "contact"
    if "." in host and not host.startswith("www."):
        return "portfolio"
    return "other"


def extract_links(text: str) -> list:
    seen = set()
    out = []
    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        url = raw if raw.startswith("http") else f"https://{raw}"
        key = url.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - 80)
        ctx = text[start:m.start()]
        label = ""
        lm = _LABEL_HINTS.search(ctx)
        if lm:
            label = lm.group(1).title()
        out.append(LinkRecord(
            url=url, category=_classify(url),
            label=label or _classify(url).title(),
        ))
    return out


def extract_company_domains(text: str) -> dict:
    mapping = {}
    for em in re.finditer(r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})", text, re.I):
        domain = em.group(2).lower()
        if domain in _IGNORED_DOMAINS or "linkedin" in domain:
            continue
        line_start = text.rfind("\n", 0, em.start()) + 1
        line = text[line_start:em.start()]
        cm = re.search(
            r"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3})", line,
        )
        if cm:
            mapping[cm.group(1).strip(" .,-")] = domain
    return mapping


async def check_url(client: httpx.AsyncClient, rec: LinkRecord) -> LinkRecord:
    try:
        r = await client.head(rec.url, headers=UA, timeout=8, follow_redirects=True)
        if r.status_code >= 400:
            r = await client.get(rec.url, headers=UA, timeout=8, follow_redirects=True)
        rec.status_code = r.status_code
        rec.live = r.status_code < 400
        if str(r.url) != rec.url:
            rec.redirect_target = str(r.url)
    except Exception as e:
        rec.live = False
        rec.error = str(e)
    return rec


async def fetch_metadata(client: httpx.AsyncClient, rec: LinkRecord) -> LinkRecord:
    if rec.category in {"contact"}:
        return rec
    try:
        r = await client.get(
            "https://api.microlink.io/",
            params={"url": rec.url}, timeout=8, headers=UA,
        )
        if r.status_code == 200:
            data = r.json().get("data", {})
            rec.title = (data.get("title") or "")[:200]
            rec.description = (data.get("description") or "")[:300]
    except Exception:
        pass
    return rec


async def detect_tech(client: httpx.AsyncClient, rec: LinkRecord) -> LinkRecord:
    if rec.category not in {"portfolio", "github"}:
        return rec
    if not rec.live:
        return rec
    try:
        from Wappalyzer import Wappalyzer, WebPage
        w = Wappalyzer.latest()
        page = await asyncio.to_thread(WebPage.new_from_url, rec.url)
        results = await asyncio.to_thread(w.analyze_with_versions_and_categories, page)
        rec.tech_stack = sorted(results.keys())[:15]
    except Exception:
        pass
    return rec


async def analyze_github(client: httpx.AsyncClient, rec: LinkRecord) -> LinkRecord:
    m = _GITHUB_RE.search(rec.url)
    if not m:
        return rec
    username = m.group(1)
    if username.lower() in {"features", "about", "pricing", "login", "signup"}:
        return rec
    try:
        r = await client.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github+json", **UA},
            timeout=8,
        )
        if r.status_code == 200:
            d = r.json()
            rec.github_meta = {
                "username": d.get("login"),
                "name": d.get("name"),
                "bio": (d.get("bio") or "")[:200],
                "public_repos": d.get("public_repos"),
                "followers": d.get("followers"),
                "created_at": d.get("created_at"),
                "blog": d.get("blog"),
            }
    except Exception:
        pass
    return rec


async def analyze_links(records, concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        async def _one(rec):
            async with sem:
                await check_url(client, rec)
                await asyncio.gather(
                    fetch_metadata(client, rec),
                    detect_tech(client, rec),
                    analyze_github(client, rec),
                    return_exceptions=True,
                )
                return rec
        return await asyncio.gather(*[_one(r) for r in records])


def analyze_resume_links(text: str) -> dict:
    records = extract_links(text)
    company_domains = extract_company_domains(text)
    if records:
        try:
            records = asyncio.run(analyze_links(records))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                records = loop.run_until_complete(analyze_links(records))
            finally:
                loop.close()

    dead = [r for r in records if r.live is False]
    live = [r for r in records if r.live is True]

    return {
        "links": [
            {
                "url": r.url, "category": r.category, "label": r.label,
                "live": r.live, "status_code": r.status_code,
                "title": r.title, "description": r.description,
                "redirect_target": r.redirect_target,
                "tech_stack": r.tech_stack, "github_meta": r.github_meta,
                "error": r.error,
            }
            for r in records
        ],
        "company_domains": company_domains,
        "summary": {
            "total": len(records), "live": len(live), "dead": len(dead),
            "by_category": {
                cat: sum(1 for r in records if r.category == cat)
                for cat in {"github", "linkedin", "portfolio", "writing", "contact", "other"}
            },
        },
    }