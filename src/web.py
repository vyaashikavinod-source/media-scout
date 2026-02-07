import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

def wiki_search(query: str, limit: int = 5):
    params = {"action":"query","list":"search","srsearch":query,"utf8":"1","format":"json"}
    r = requests.get(WIKI_API, params=params, timeout=20)
    r.raise_for_status()
    hits = r.json()["query"]["search"][:limit]
    return [h["title"] for h in hits]

def wiki_fetch_summary(title: str):
    url = WIKI_REST_SUMMARY.format(quote(title, safe=""))
    r = requests.get(url, timeout=20, headers={"User-Agent":"llm-playground/1.0"})
    r.raise_for_status()
    j = r.json()
    text = j.get("extract") or ""
    page_url = (j.get("content_urls", {}).get("desktop", {}) or {}).get("page", "")
    thumb = (j.get("thumbnail") or {}).get("source")
    return {"title": j.get("title", title),
            "url": page_url or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ','_'))}",
            "text": text,
            "image": thumb}

def openlibrary_search(query: str, limit: int = 5):
    url = "https://openlibrary.org/search.json"
    r = requests.get(url, params={"q": query, "limit": limit}, timeout=20)
    r.raise_for_status()
    return r.json().get("docs", [])[:limit]

def openlibrary_fetch_book(doc: dict):
    title = doc.get("title") or "Unknown title"
    author = (doc.get("author_name") or ["Unknown author"])[0]
    year = doc.get("first_publish_year")
    key = doc.get("key")
    work_url = f"https://openlibrary.org{key}" if key else "https://openlibrary.org"
    cover_id = doc.get("cover_i")
    cover = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

    desc = ""
    if key:
        try:
            w = requests.get(f"https://openlibrary.org{key}.json", timeout=20)
            if w.ok:
                wj = w.json()
                d = wj.get("description")
                if isinstance(d, dict):
                    desc = d.get("value","")
                elif isinstance(d, str):
                    desc = d
        except Exception:
            pass

    text = desc.strip()
    if not text:
        parts = [f"{title}", f"Author: {author}"]
        if year:
            parts.append(f"First published: {year}")
        text = ". ".join(parts) + "."

    return {"title": f"{title} — {author}" + (f" ({year})" if year else ""),
            "url": work_url,
            "text": text,
            "image": cover}

def fetch_url_text(url: str, max_chars: int = 20000):
    r = requests.get(url, timeout=25, headers={"User-Agent":"llm-playground/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+"," ",text).strip()
    return text[:max_chars]
