import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


def _clean_refs(text: str) -> str:
    # remove Wikipedia-style reference markers like [1], [12], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _abs_url(base: str, maybe: Optional[str]) -> Optional[str]:
    if not maybe:
        return None
    if maybe.startswith("http"):
        return maybe
    if maybe.startswith("//"):
        return "https:" + maybe
    if maybe.startswith("/"):
        return base.rstrip("/") + maybe
    return maybe


def _sanitize_url(url: str) -> str:
    url = url.strip()
    # common copy-paste junk
    url = url.strip(")];.,")
    return url


@dataclass
class SourceDoc:
    title: str
    url: str
    image_url: Optional[str]
    summary: str
    plot: str
    full_text: str


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    })
    return s


def wiki_search(session: requests.Session, query: str, limit: int = 5) -> List[Tuple[str, str]]:
    # MediaWiki API (works well when User-Agent is set)
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": query,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    }
    r = session.get(api, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    titles = data[1]
    urls = data[3]
    out = []
    for t, u in zip(titles, urls):
        out.append((t, u))
    return out


def wiki_fetch(session: requests.Session, title: str) -> SourceDoc:
    page_url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
    r = session.get(page_url, timeout=25)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    h1 = soup.find("h1", id="firstHeading")
    page_title = h1.get_text(" ", strip=True) if h1 else title

    # infobox image (best effort)
    img = soup.select_one(".infobox img")
    image_url = _abs_url("https://en.wikipedia.org", img.get("src") if img else None)

    content = soup.select_one(".mw-parser-output")
    if not content:
        raise RuntimeError("Wikipedia page structure not found.")

    # lead summary: paragraphs before first h2
    lead_parts = []
    for el in content.children:
        name = getattr(el, "name", None)
        if name == "h2":
            break
        if name == "p":
            txt = el.get_text(" ", strip=True)
            txt = _clean_refs(txt)
            if txt:
                lead_parts.append(txt)
    summary = " ".join(lead_parts)[:6000].strip()

    # plot section: try common headings
    plot_text = ""
    for plot_id in ("Plot", "Synopsis", "Premise", "Summary"):
        head = soup.find(id=plot_id)
        if head:
            # move up to the heading tag
            heading = head.parent
            chunks = []
            for sib in heading.find_all_next():
                if sib.name in ("h2", "h3") and sib.find(id=True):
                    break
                if sib.name == "p":
                    t = _clean_refs(sib.get_text(" ", strip=True))
                    if t:
                        chunks.append(t)
                if len(" ".join(chunks)) > 8000:
                    break
            plot_text = " ".join(chunks).strip()
            if plot_text:
                break

    # full text for evidence retrieval (use lead + plot + some extra paragraphs)
    extra = []
    pcount = 0
    for p in content.find_all("p"):
        t = _clean_refs(p.get_text(" ", strip=True))
        if t:
            extra.append(t)
            pcount += 1
        if pcount >= 40:
            break

    full_text = "\n\n".join(extra)
    return SourceDoc(
        title=page_title,
        url=page_url,
        image_url=image_url,
        summary=summary,
        plot=plot_text,
        full_text=full_text,
    )


def openlibrary_search(session: requests.Session, query: str, limit: int = 5) -> List[dict]:
    # Prefer title search to avoid weird mismatches
    url = "https://openlibrary.org/search.json"
    params = {"title": query, "limit": limit}
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("docs", [])


def openlibrary_fetch(session: requests.Session, doc: dict) -> SourceDoc:
    key = doc.get("key")  # like /works/OL...
    if not key:
        raise RuntimeError("Open Library result missing key.")
    work_url = "https://openlibrary.org" + key
    r = session.get(work_url + ".json", timeout=20)
    r.raise_for_status()
    w = r.json()

    title = w.get("title") or doc.get("title") or "Untitled"
    desc = w.get("description", "")
    if isinstance(desc, dict):
        desc = desc.get("value", "")
    summary = (desc or doc.get("first_sentence") or "")
    summary = str(summary).strip()

    cover_id = None
    covers = w.get("covers") or []
    if covers:
        cover_id = covers[0]
    elif doc.get("cover_i"):
        cover_id = doc.get("cover_i")

    image_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

    # Open Library often doesn't have a "plot"; treat description as summary and keep plot empty
    full_text = summary
    return SourceDoc(
        title=title,
        url=work_url,
        image_url=image_url,
        summary=summary,
        plot="",
        full_text=full_text,
    )


def url_fetch(session: requests.Session, url: str, max_chars: int = 20000) -> SourceDoc:
    url = _sanitize_url(url)
    r = session.get(url, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else url

    main = soup.find("article") or soup.find("main") or soup.body
    ps = main.find_all("p") if main else []
    text = "\n\n".join([p.get_text(" ", strip=True) for p in ps])
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:max_chars]

    return SourceDoc(
        title=title,
        url=url,
        image_url=None,
        summary=text[:1200],
        plot="",
        full_text=text,
    )
