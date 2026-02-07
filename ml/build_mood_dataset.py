import os
import json
import time
import random
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": "MediaScout/1.0 (local)",
    "Accept": "application/json",
}

MOODS = {
    # mood_name: TMDB genre ids used to fetch training examples
    "feel_good": [35, 10751],     # Comedy, Family
    "romantic": [10749],          # Romance
    "spooky": [27, 53],           # Horror, Thriller
    "action": [28, 12],           # Action, Adventure
    "mind_bending": [878, 9648],  # Sci-Fi, Mystery
}

def load_secrets():
    # scripts are run from repo root; read Streamlit secrets file if present
    secrets_path = Path(".streamlit") / "secrets.toml"
    out = {}
    if secrets_path.exists():
        # Python 3.12 has tomllib
        import tomllib
        with open(secrets_path, "rb") as f:
            out = tomllib.load(f)
    # env overrides
    out["TMDB_V3_API_KEY"] = os.getenv("TMDB_V3_API_KEY", out.get("TMDB_V3_API_KEY", ""))
    out["TMDB_V4_READ_TOKEN"] = os.getenv("TMDB_V4_READ_TOKEN", out.get("TMDB_V4_READ_TOKEN", ""))
    return out

def tmdb_headers(v4_token: str):
    h = dict(HEADERS)
    if v4_token:
        h["Authorization"] = f"Bearer {v4_token}"
    return h

def tmdb_get(path: str, params: dict, v3_key: str, v4_token: str):
    url = f"https://api.themoviedb.org/3/{path.lstrip('/')}"
    p = dict(params or {})
    # if no v4 token, use v3 key
    if not v4_token:
        if not v3_key:
            raise RuntimeError("Missing TMDB creds. Add TMDB_V4_READ_TOKEN or TMDB_V3_API_KEY in .streamlit/secrets.toml")
        p["api_key"] = v3_key
    r = requests.get(url, params=p, headers=tmdb_headers(v4_token), timeout=20)
    r.raise_for_status()
    return r.json()

def build(per_mood=500, pages=8, seed=7):
    random.seed(seed)
    secrets = load_secrets()
    v3 = (secrets.get("TMDB_V3_API_KEY") or "").strip()
    v4 = (secrets.get("TMDB_V4_READ_TOKEN") or "").strip()

    out_path = Path("data") / "mood_dataset.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    rows = []

    for mood, genres in MOODS.items():
        collected = 0
        for page in range(1, pages + 1):
            if collected >= per_mood:
                break

            j = tmdb_get(
                "discover/movie",
                params={
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "with_genres": ",".join(map(str, genres)),
                    "page": page,
                    "language": "en-US",
                },
                v3_key=v3,
                v4_token=v4,
            )
            results = j.get("results", []) or []

            for m in results:
                if collected >= per_mood:
                    break
                mid = m.get("id")
                overview = (m.get("overview") or "").strip()
                title = (m.get("title") or "").strip()
                if not mid or mid in seen_ids:
                    continue
                if len(overview) < 40:
                    continue
                seen_ids.add(mid)
                rows.append({
                    "id": mid,
                    "title": title,
                    "overview": overview,
                    "label": mood,
                })
                collected += 1

            time.sleep(0.25)  # be gentle on rate limits

        print(f"✅ {mood}: {collected} examples")

    random.shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ wrote {out_path} with {len(rows)} rows")

if __name__ == "__main__":
    build()
