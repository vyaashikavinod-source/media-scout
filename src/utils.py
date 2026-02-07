import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse, quote

import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HEADERS = {
    "User-Agent": "project-a-media-research-assistant/1.0 (educational; no contact)"
}


def clean_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""

    # Remove surrounding <> sometimes copied from markdown
    if u.startswith("<") and u.endswith(">"):
        u = u[1:-1].strip()

    # Ensure scheme
    p = urlparse(u)
    if not p.scheme:
        u = "https://" + u
        p = urlparse(u)

    # Fix extra closing parens at end (common with Wikipedia titles)
    opens = u.count("(")
    closes = u.count(")")
    while closes > opens and u.endswith(")"):
        u = u[:-1]
        closes -= 1

    # Trim trailing punctuation (but not slash)
    u = u.rstrip(" \t\r\n.,;]")

    return u


def _get(url: str, timeout: int = 25) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def fetch_url_text(url: str, max_chars: int = 20000) -> str:
    url = clean_url(url)
    r = _get(url)
    soup = BeautifulSoup(r.text, "lxml")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()

    # Wikipedia: focus content
    content = soup.select_one("div.mw-parser-output") or soup.body or soup

    text = content.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def wiki_search_html(query: str, limit: int = 5) -> List[str]:
    # Use normal Wikipedia search page (not API)
    url = "https://en.wikipedia.org/w/index.php"
    r = requests.get(url, params={"search": query}, headers=HEADERS, timeout=20, allow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Direct hit
    heading = soup.select_one("#firstHeading")
    if heading and heading.get_text(strip=True) and "/wiki/" in r.url:
        return [heading.get_text(strip=True)]

    links = soup.select(".mw-search-result-heading a")[:limit]
    titles = []
    for a in links:
        t = a.get("title") or a.get_text(strip=True)
        if t:
            titles.append(t)
    return titles


def wiki_title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def openlibrary_search(query: str, limit: int = 5) -> List[Dict]:
    url = "https://openlibrary.org/search.json"
    r = requests.get(url, params={"q": query, "limit": limit}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    docs = (r.json() or {}).get("docs", [])[:limit]

    out = []
    for d in docs:
        title = d.get("title") or ""
        author = (d.get("author_name") or [""])[0]
        year = d.get("first_publish_year")
        key = d.get("key")  # /works/...
        cover = d.get("cover_i")
        work_url = f"https://openlibrary.org{key}" if key else ""
        cover_url = f"https://covers.openlibrary.org/b/id/{cover}-L.jpg" if cover else None
        out.append({
            "title": title,
            "author": author,
            "year": year,
            "url": work_url,
            "image": cover_url,
        })
    return out


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


@dataclass
class Chunk:
    source_title: str
    source_url: str
    text: str


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.X = self.vectorizer.fit_transform([c.text for c in chunks]) if chunks else None

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        if not self.chunks:
            return []
        if not query.strip():
            query = "summary"

        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).ravel()

        # If query is generic and scores are all ~0, just return early chunks
        if sims.max() < 1e-6:
            out = []
            for i, c in enumerate(self.chunks[:k]):
                out.append((0.0, c))
            return out

        idxs = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.chunks[i]) for i in idxs]


def extractive_bullets(chunks: List[Chunk], bullets: int = 5) -> List[str]:
    # Very simple: take first sentences across top chunks, de-dup
    import re
    seen = set()
    out = []
    for c in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", c.text)
        for s in sentences:
            s = s.strip()
            if len(s) < 40:
                continue
            key = s[:120].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= bullets:
                return out
    return out
