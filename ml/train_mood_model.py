import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = Path("data") / "mood_dataset.jsonl"
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python -m ml.build_mood_dataset")
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    rows = load_rows()
    texts = [(r["title"] + ". " + r["overview"]).strip() for r in rows]
    labels = [r["label"] for r in rows]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=40000,
            ngram_range=(1, 2),
        )),
        ("lr", LogisticRegression(
            max_iter=400,
            n_jobs=None,
            class_weight="balanced",
        )),
    ])

    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    acc = accuracy_score(y_val, pred)

    print("\n==============================")
    print(f"✅ Validation accuracy: {acc:.3f}")
    print("==============================\n")
    print(classification_report(y_val, pred))

    joblib.dump(clf, OUT_DIR / "mood_model.joblib")

    metrics = {
        "val_accuracy": float(acc),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "labels": sorted(list(set(labels))),
    }
    with open(OUT_DIR / "mood_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ saved artifacts/mood_model.joblib")
    print(f"✅ saved artifacts/mood_metrics.json")

if __name__ == "__main__":
    main()
