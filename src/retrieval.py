from dataclasses import dataclass
from typing import List, Tuple
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_chars - overlap)
    return chunks


def top_k_passages(question: str, passages: List[str], k: int = 5) -> List[Tuple[float, str]]:
    if not passages:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vec.fit_transform(passages)
    q = vec.transform([question])
    scores = (X @ q.T).toarray().ravel()
    idx = np.argsort(-scores)[:k]
    out = []
    for i in idx:
        out.append((float(scores[i]), passages[int(i)]))
    return out


def bullet_summary(text: str, n: int = 5) -> List[str]:
    # cheap extractive bullets: first n sentences
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    bullets = []
    for s in sents:
        s = s.strip()
        if s and len(s) > 30:
            bullets.append(s)
        if len(bullets) >= n:
            break
    return bullets
