import os
import re
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

# Optional (only used if your trained model exists)
try:
    import joblib
except Exception:
    joblib = None


# -----------------------------
# Page config + CSS
# -----------------------------
st.set_page_config(page_title="Media Scout", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
.smallcap { opacity: 0.75; font-size: 0.9rem; }
hr { opacity: 0.25; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 14px;
  background: rgba(255,255,255,0.03);
  transition: transform 160ms ease, box-shadow 160ms ease;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}
.poster {
  border-radius: 14px;
  overflow: hidden;
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  margin-right: 6px;
  margin-bottom: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

HEADERS = {
    "User-Agent": "MediaScout/1.0 (streamlit; contact: demo@example.com)",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}

APP_DIR = Path(__file__).parent
ARTIFACTS_DIR = APP_DIR / "artifacts"


# -----------------------------
# BEFORE WIDGETS: memory / state init
# -----------------------------
def init_state():
    defaults = {
        "page": "Media Scout",
        "mode": "Movie",
        "country": "US",
        "genre": "Any",
        "mood": "Chill / Cozy",
        "fav_movie": "Kill Bill",
        "query": "Titanic",
        "bullets": 2,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# Load saved settings from URL (shareable memory)
# Example URL: ?country=GB&mood=Dark&fav=Kill%20Bill&genre=Action&page=Mood%20Picks
try:
    qp = st.query_params
    if "page" in qp:
        st.session_state["page"] = qp["page"]
    if "country" in qp:
        st.session_state["country"] = qp["country"]
    if "mood" in qp:
        st.session_state["mood"] = qp["mood"]
    if "fav" in qp:
        st.session_state["fav_movie"] = qp["fav"]
    if "genre" in qp:
        st.session_state["genre"] = qp["genre"]
except Exception:
    pass


# -----------------------------
# Helpers
# -----------------------------
def clean_query(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def similarity(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def make_bullets(text: str, n: int = 2) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    # Sentence split (simple)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 35]
    if not sents:
        # fallback: first chunk
        return [text[:180].strip() + ("…" if len(text) > 180 else "")]
    # Prefer first few meaningful sentences
    out = []
    for s in sents:
        if len(out) >= n:
            break
        if s not in out:
            out.append(s)
    return out[:n]


# -----------------------------
# Secrets (TMDB)
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.environ.get(name, default)


TMDB_V3_API_KEY = get_secret("TMDB_V3_API_KEY", "")
TMDB_V4_READ_TOKEN = get_secret("TMDB_V4_READ_TOKEN", "")
DEFAULT_REGION = get_secret("DEFAULT_REGION", "US")

TMDB_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP = "https://image.tmdb.org/t/p/w780"


def tmdb_auth_headers() -> Dict[str, str]:
    if TMDB_V4_READ_TOKEN:
        return {"Authorization": f"Bearer {TMDB_V4_READ_TOKEN}"}
    return {}


def tmdb_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"https://api.themoviedb.org/3{path}"
    params = params or {}
    headers = {**HEADERS, **tmdb_auth_headers()}

    # If no v4 token, fallback to v3 key param
    if not TMDB_V4_READ_TOKEN:
        if not TMDB_V3_API_KEY:
            raise RuntimeError("TMDB key missing. Add TMDB_V3_API_KEY or TMDB_V4_READ_TOKEN to secrets.toml.")
        params["api_key"] = TMDB_V3_API_KEY

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json() or {}


# -----------------------------
# Optional: load your trained mood model (joblib)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_mood_model():
    p = ARTIFACTS_DIR / "mood_model.joblib"
    if joblib is None or not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


MOOD_MODEL = load_mood_model()


def predict_mood_from_text(text: str) -> Optional[Tuple[str, float]]:
    """Returns (label, confidence) if model exists."""
    if MOOD_MODEL is None:
        return None
    text = (text or "").strip()
    if not text:
        return None
    try:
        proba = MOOD_MODEL.predict_proba([text])[0]
        idx = int(proba.argmax())
        label = MOOD_MODEL.classes_[idx]
        conf = float(proba[idx])
        return label, conf
    except Exception:
        return None


# -----------------------------
# APIs
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_genres() -> Dict[str, int]:
    j = tmdb_get("/genre/movie/list", params={"language": "en-US"})
    items = j.get("genres", []) or []
    name_to_id = {g["name"]: int(g["id"]) for g in items if g.get("name") and g.get("id")}
    return name_to_id


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_search_movie(query: str, limit: int = 10) -> List[dict]:
    query = clean_query(query)
    if not query:
        return []
    j = tmdb_get("/search/movie", params={"query": query, "include_adult": "false", "language": "en-US", "page": 1})
    res = j.get("results", []) or []
    return res[:limit]


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_movie_details(movie_id: int) -> dict:
    return tmdb_get(f"/movie/{movie_id}", params={"language": "en-US"})


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_watch_providers(movie_id: int) -> dict:
    j = tmdb_get(f"/movie/{movie_id}/watch/providers", params={})
    return (j.get("results", {}) or {})


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_movie_videos(movie_id: int) -> List[dict]:
    j = tmdb_get(f"/movie/{movie_id}/videos", params={"language": "en-US"})
    return (j.get("results", []) or [])


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_recommendations(movie_id: int, limit: int = 12) -> List[dict]:
    j = tmdb_get(f"/movie/{movie_id}/recommendations", params={"language": "en-US", "page": 1})
    return (j.get("results", []) or [])[:limit]


@st.cache_data(show_spinner=False, ttl=3600)
def tmdb_discover_movies(
    region: str,
    genre_id: Optional[int],
    mood: str,
    limit: int = 12,
) -> List[dict]:
    """
    Mood -> discover filters (lightweight + fast).
    """
    region = (region or DEFAULT_REGION or "US").upper()

    mood_map = {
        "Chill / Cozy": {"with_genres": "35,10751,10749"},          # Comedy, Family, Romance
        "Hype": {"with_genres": "28,12,16"},                        # Action, Adventure, Animation
        "Dark": {"with_genres": "53,27,80"},                        # Thriller, Horror, Crime
        "Romantic": {"with_genres": "10749,18"},                    # Romance, Drama
        "Funny": {"with_genres": "35"},                             # Comedy
        "Mind-bending": {"with_genres": "878,9648,53"},             # Sci-Fi, Mystery, Thriller
    }
    mm = mood_map.get(mood, {})

    params = {
        "language": "en-US",
        "page": 1,
        "region": region,
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "vote_count.gte": 250,
    }

    # Combine mood genres + selected genre if any
    if mm.get("with_genres"):
        params["with_genres"] = mm["with_genres"]
    if genre_id:
        if "with_genres" in params and params["with_genres"]:
            params["with_genres"] = f"{params['with_genres']},{genre_id}"
        else:
            params["with_genres"] = str(genre_id)

    j = tmdb_get("/discover/movie", params=params)
    return (j.get("results", []) or [])[:limit]


# iTunes Search API (pricing when available)
@st.cache_data(show_spinner=False, ttl=3600)
def itunes_movie_offers(query: str, country: str = "US", limit: int = 10) -> List[dict]:
    query = clean_query(query)
    if not query:
        return []
    url = "https://itunes.apple.com/search"
    params = {
        "term": query,
        "media": "movie",
        "entity": "movie",
        "limit": limit,
        "country": (country or "US").lower(),
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    j = r.json() or {}
    results = j.get("results", []) or []
    out = []
    for x in results:
        out.append(
            {
                "trackName": x.get("trackName"),
                "releaseDate": x.get("releaseDate"),
                "trackViewUrl": x.get("trackViewUrl"),
                "artworkUrl100": x.get("artworkUrl100"),
                "trackPrice": x.get("trackPrice"),
                "trackRentalPrice": x.get("trackRentalPrice"),
                "trackHdPrice": x.get("trackHdPrice"),
                "trackHdRentalPrice": x.get("trackHdRentalPrice"),
                "currency": x.get("currency"),
            }
        )
    return out


# Open Library (books)
@st.cache_data(show_spinner=False, ttl=3600)
def openlibrary_search(title: str, limit: int = 10) -> List[dict]:
    title = clean_query(title)
    if not title:
        return []
    url = "https://openlibrary.org/search.json"
    params = {"title": title, "limit": limit}
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    j = r.json() or {}
    docs = j.get("docs", []) or []
    out = []
    for d in docs:
        key = d.get("key") or ""
        olid = key.replace("/works/", "") if key else ""
        out.append(
            {
                "title": d.get("title") or "",
                "author": (d.get("author_name") or [""])[0],
                "year": d.get("first_publish_year"),
                "olid": olid,
                "cover_i": d.get("cover_i"),
                "url": f"https://openlibrary.org{key}" if key else "",
            }
        )
    return out


def pick_best_youtube_trailer(videos: List[dict]) -> Optional[str]:
    """
    Return YouTube URL for best trailer (prefer official Trailer).
    """
    yt = [v for v in videos if (v.get("site") == "YouTube" and v.get("key"))]
    if not yt:
        return None

    def score(v):
        name = (v.get("name") or "").lower()
        t = (v.get("type") or "").lower()
        official = bool(v.get("official"))
        s = 0
        if "trailer" in t:
            s += 50
        if "official" in name:
            s += 20
        if official:
            s += 10
        if "teaser" in t:
            s += 5
        return s

    best = sorted(yt, key=score, reverse=True)[0]
    return f"https://www.youtube.com/watch?v={best['key']}"


def provider_names(block) -> List[str]:
    if not block:
        return []
    return [p.get("provider_name") for p in (block or []) if p.get("provider_name")]


def poster_url(path: Optional[str]) -> str:
    if not path:
        return ""
    return f"{TMDB_IMG}{path}"


# -----------------------------
# UI Components
# -----------------------------
def render_movie_grid(movies: List[dict], cols: int = 3, key_prefix: str = "grid"):
    if not movies:
        st.info("No results.")
        return

    rows = math.ceil(len(movies) / cols)
    idx = 0
    for r in range(rows):
        c = st.columns(cols, gap="large")
        for j in range(cols):
            if idx >= len(movies):
                break
            m = movies[idx]
            idx += 1

            title = m.get("title") or m.get("name") or "Untitled"
            year = (m.get("release_date") or "")[:4]
            rating = m.get("vote_average")
            mid = m.get("id")

            with c[j]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                p = poster_url(m.get("poster_path"))
                if p:
                    st.image(p, use_container_width=True)
                st.markdown(f"**{title}**" + (f" ({year})" if year else ""))
                meta = []
                if rating is not None:
                    meta.append(f"⭐ {rating:.1f}")
                if year:
                    meta.append(f"📅 {year}")
                if meta:
                    st.caption(" • ".join(meta))

                if mid:
                    # TMDB link button (nice + reliable)
                    st.link_button("Open on TMDB", f"https://www.themoviedb.org/movie/{mid}", use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Sidebar / Controls
# -----------------------------
st.title("🎬 Media Scout")
st.caption("Search movies & books, watch trailers, see where-to-watch, and get mood-based picks + recommendations.")

with st.sidebar:
    st.subheader("Navigation")
    st.selectbox("Page", ["Media Scout", "Mood Picks"], key="page")

    st.divider()
    st.subheader("Settings")
    st.selectbox("Region", ["US", "IN", "GB", "CA", "AU"], key="country")
    st.text_input("Favorite movie", key="fav_movie")

    st.selectbox(
        "Mood",
        ["Chill / Cozy", "Hype", "Dark", "Romantic", "Funny", "Mind-bending"],
        key="mood",
    )

    # Genres (TMDB)
    try:
        genre_map = tmdb_genres()
        genre_names = ["Any"] + sorted(list(genre_map.keys()))
    except Exception:
        genre_map = {}
        genre_names = ["Any"]

    st.selectbox("Genre", genre_names, key="genre")

    st.divider()
    st.subheader("Share")
    if st.button("Save/share my settings", use_container_width=True):
        try:
            st.query_params["page"] = st.session_state["page"]
            st.query_params["country"] = st.session_state["country"]
            st.query_params["mood"] = st.session_state["mood"]
            st.query_params["fav"] = st.session_state["fav_movie"]
            st.query_params["genre"] = st.session_state["genre"]
            st.success("Saved in the URL. Copy the browser link now.")
        except Exception:
            st.warning("Could not write query params in this environment.")

    if st.button("Reset", use_container_width=True):
        for k in ["page", "mode", "country", "genre", "mood", "fav_movie", "query", "bullets"]:
            if k in st.session_state:
                del st.session_state[k]
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()


# -----------------------------
# Main Pages
# -----------------------------
page = st.session_state["page"]
country = st.session_state["country"]
mood = st.session_state["mood"]
fav_movie = clean_query(st.session_state["fav_movie"])
genre_name = st.session_state["genre"]
genre_id = None

if genre_name != "Any":
    genre_id = genre_map.get(genre_name)

if page == "Media Scout":
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Search")
        st.selectbox("Type", ["Movie", "Book"], key="mode")
        st.slider("Bullet points", 1, 4, key="bullets")
        st.text_input("Title", key="query")

        st.caption("Tip: keep it short (example: Titanic, Interstellar, Friends).")

    with right:
        mode = st.session_state["mode"]
        q = clean_query(st.session_state["query"])
        bullets_n = int(st.session_state["bullets"])

        if not q:
            st.info("Type a title to search.")
            st.stop()

        if mode == "Movie":
            st.subheader("Pick the correct match")
            try:
                results = tmdb_search_movie(q, limit=10)
            except Exception as e:
                st.error(f"TMDB search failed: {e}")
                st.stop()

            if not results:
                st.warning("No results. Try a simpler title.")
                st.stop()

            labels = []
            for r in results:
                title = r.get("title", "")
                year = (r.get("release_date") or "")[:4]
                labels.append(f"{title} ({year})" if year else title)

            pick = st.selectbox("Results", labels, key="movie_pick")
            chosen = results[labels.index(pick)]
            movie_id = int(chosen["id"])

            # Details
            d = tmdb_movie_details(movie_id)
            title = d.get("title") or chosen.get("title") or "Untitled"
            overview = d.get("overview") or chosen.get("overview") or ""
            release_date = d.get("release_date") or chosen.get("release_date") or ""
            year = release_date[:4] if release_date else ""
            poster = poster_url(d.get("poster_path") or chosen.get("poster_path"))
            backdrop = d.get("backdrop_path")
            genres = [g.get("name") for g in (d.get("genres") or []) if g.get("name")]
            vote = d.get("vote_average")

            c1, c2 = st.columns([1, 1], gap="large")

            with c1:
                st.markdown("### Details")
                if poster:
                    st.image(poster, width=260)
                tags = []
                if year:
                    tags.append(f"<span class='badge'>📅 {year}</span>")
                if vote is not None:
                    tags.append(f"<span class='badge'>⭐ {vote:.1f}</span>")
                if genres:
                    tags.append(f"<span class='badge'>🎭 {', '.join(genres[:3])}</span>")
                if tags:
                    st.markdown(" ".join(tags), unsafe_allow_html=True)

                st.link_button("Open on TMDB", f"https://www.themoviedb.org/movie/{movie_id}")

                st.markdown("### Quick synopsis")
                for b in make_bullets(overview, bullets_n):
                    st.write(f"• {b}")

                # Mood prediction from synopsis (if your model exists)
                pred = predict_mood_from_text(overview)
                if pred:
                    label, conf = pred
                    st.caption(f"Predicted mood from synopsis: **{label}** (confidence {conf:.2f})")

            with c2:
                st.markdown("### Trailer")
                try:
                    vids = tmdb_movie_videos(movie_id)
                    trailer = pick_best_youtube_trailer(vids)
                except Exception:
                    trailer = None

                if trailer:
                    st.video(trailer)
                else:
                    st.info("No trailer found on TMDB for this title.")

                st.markdown("### Where to watch")
                # TMDB providers
                try:
                    prov = tmdb_watch_providers(movie_id)
                    region_block = (prov.get(country.upper()) or {})
                    flatrate = provider_names(region_block.get("flatrate"))
                    rent = provider_names(region_block.get("rent"))
                    buy = provider_names(region_block.get("buy"))

                    if flatrate:
                        st.write("Streaming")
                        st.write(", ".join(flatrate))
                    if rent:
                        st.write("Rent")
                        st.write(", ".join(rent))
                    if buy:
                        st.write("Buy")
                        st.write(", ".join(buy))

                    if not (flatrate or rent or buy):
                        st.caption("No provider list found for this region.")
                except Exception as e:
                    st.caption(f"Provider lookup failed: {e}")

                st.markdown("### Apple TV pricing (when available)")
                try:
                    offers = itunes_movie_offers(title, country=country, limit=10)
                except Exception as e:
                    offers = []
                    st.caption(f"iTunes lookup failed: {e}")

                if offers:
                    offers_sorted = sorted(offers, key=lambda x: similarity(x.get("trackName", ""), title), reverse=True)
                    best = offers_sorted[0]
                    cur = best.get("currency") or ""
                    rent_sd = best.get("trackRentalPrice")
                    rent_hd = best.get("trackHdRentalPrice")
                    buy_sd = best.get("trackPrice")
                    buy_hd = best.get("trackHdPrice")

                    rows = []
                    if rent_sd is not None:
                        rows.append(("Rent (SD)", f"{rent_sd} {cur}"))
                    if rent_hd is not None:
                        rows.append(("Rent (HD)", f"{rent_hd} {cur}"))
                    if buy_sd is not None:
                        rows.append(("Buy (SD)", f"{buy_sd} {cur}"))
                    if buy_hd is not None:
                        rows.append(("Buy (HD)", f"{buy_hd} {cur}"))

                    if rows:
                        for k, v in rows:
                            st.write(f"- {k}: {v}")
                    else:
                        st.write("- Price not returned for this item/region.")

                    link = best.get("trackViewUrl") or ""
                    if link:
                        st.link_button("Open on Apple TV", link, use_container_width=True)
                else:
                    st.write("- Not found on iTunes for this region (or no pricing returned).")

            st.divider()
            st.subheader("Recommended For You (based on this movie)")
            try:
                recs = tmdb_recommendations(movie_id, limit=9)
                render_movie_grid(recs, cols=3, key_prefix="recs_from_movie")
            except Exception as e:
                st.caption(f"Recommendations failed: {e}")

        else:
            # Book mode
            st.subheader("Pick the correct match")
            try:
                results = openlibrary_search(q, limit=10)
            except Exception as e:
                st.error(f"Open Library search failed: {e}")
                st.stop()

            if not results:
                st.warning("No results. Try a simpler title.")
                st.stop()

            labels = [f"{r['title']} — {r['author']} ({r['year']})" for r in results]
            pick = st.selectbox("Results", labels, key="book_pick")
            chosen = results[labels.index(pick)]

            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("### Details")
                if chosen.get("cover_i"):
                    st.image(f"https://covers.openlibrary.org/b/id/{chosen['cover_i']}-L.jpg", width=260)
                st.write(f"**Title:** {chosen.get('title','')}")
                st.write(f"**Author:** {chosen.get('author','')}")
                st.write(f"**First published:** {chosen.get('year','')}")
                if chosen.get("url"):
                    st.link_button("Open on Open Library", chosen["url"], use_container_width=True)

            with c2:
                st.markdown("### Quick synopsis")
                base = f"{chosen.get('title','')}. By {chosen.get('author','')}. First published {chosen.get('year','')}."
                for b in make_bullets(base, bullets_n):
                    st.write(f"• {b}")

else:
    # Mood Picks page
    st.subheader("Mood Picks")

    st.write(
        f"**Region:** {country}  |  **Mood:** {mood}  |  **Genre:** {genre_name}  |  **Favorite:** {fav_movie or '—'}"
    )
    st.caption("Two clean sections: Your Picks (mood + region + genre) and Recommended For You (favorite movie).")

    st.divider()
    st.markdown("## Your Picks")
    try:
        picks = tmdb_discover_movies(region=country, genre_id=genre_id, mood=mood, limit=12)
        render_movie_grid(picks[:6], cols=3, key_prefix="your_picks")
    except Exception as e:
        st.error(f"Could not load mood picks: {e}")

    st.divider()
    st.markdown("## Recommended For You")
    if not fav_movie:
        st.info("Type your favorite movie in the sidebar to see recommendations here.")
    else:
        try:
            fav_res = tmdb_search_movie(fav_movie, limit=5)
            if not fav_res:
                st.warning("Could not find your favorite movie on TMDB. Try a simpler title.")
            else:
                # choose best match by similarity
                best = sorted(fav_res, key=lambda x: similarity(x.get("title", ""), fav_movie), reverse=True)[0]
                fav_id = int(best["id"])
                recs = tmdb_recommendations(fav_id, limit=12)
                render_movie_grid(recs[:6], cols=3, key_prefix="fav_recs")
        except Exception as e:
            st.error(f"Could not load recommendations: {e}")
