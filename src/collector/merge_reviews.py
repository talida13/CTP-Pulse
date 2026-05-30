"""
merge_reviews.py
----------------
Combină cele 3 fișiere JSON (facebook.json, gmaps_ctp.json, gmaps_ratp.json)
într-un singur CSV curat și deduplicat.
Owner: Talida Caraman
Input:  data/raw/facebook.json, data/raw/gmaps_ctp.json, data/raw/gmaps_ratp.json
Output: data/processed/reviews_merged.csv
TODO:
- [ ] Parser pentru structura Facebook (text, date, likesCount, isRecommended)
- [ ] Parser pentru structura Google Maps (text, stars, name, reviewUrl)
- [ ] Deduplicare prin hash MD5 pe text
- [ ] Filtrare recenzii fără text (text: null)
- [ ] Salvare CSV cu coloane uniforme: source, location, text, rating, author, review_date, url
"""
import os
import json
import hashlib
import argparse
import pandas as pd
from datetime import datetime

# ─── Parsere ──────────────────────────────────────────────────────────────────

def parse_facebook(item: dict) -> dict | None:
    text = (item.get("text") or "").strip()
    if not text:
        return None
    return {
        "source":         "facebook",
        "location":       item.get("facebookUrl", "CTP Iași Facebook"),
        "text":           text,
        "rating":         None,
        "is_recommended": item.get("isRecommended"),
        "likes":          item.get("likesCount", 0),
        "author":         item.get("authorName") or item.get("userName", ""),
        "review_date":    item.get("date", ""),
        "url":            item.get("postUrl") or item.get("url", ""),
    }

def parse_gmaps(item: dict, location_name: str) -> dict | None:
    text = (item.get("text") or "").strip()
    if not text:
        return None
    return {
        "source":         "google_maps",
        "location":       item.get("title") or location_name,
        "text":           text,
        "rating":         item.get("stars"),
        "is_recommended": None,
        "likes":          None,
        "author":         item.get("name", ""),
        "review_date":    item.get("publishedAtDate") or item.get("date", ""),
        "url":            item.get("reviewUrl") or item.get("url", ""),
    }

# ─── Helpers ──────────────────────────────────────────────────────────────────

def text_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"  ⚠ Fișier negăsit, sărit: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Apify poate returna {"items": [...]} sau direct o listă
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    print(f"  {os.path.basename(path)}: {len(data)} items")
    return data

# ─── Main ─────────────────────────────────────────────────────────────────────

def merge(data_dir: str = "data/raw") -> pd.DataFrame:
    sources = [
        (os.path.join(data_dir, "facebook.json"),    "facebook",  "CTP Iași Facebook"),
        (os.path.join(data_dir, "gmaps_ctp.json"),   "gmaps",     "CTP Iași HQ"),
        (os.path.join(data_dir, "gmaps_ratp.json"),  "gmaps",     "RATP Iași"),
    ]

    rows = []
    seen = set()

    for path, kind, location_name in sources:
        items = load_json(path)
        added = skipped_empty = skipped_dup = 0

        for item in items:
            parsed = parse_facebook(item) if kind == "facebook" else parse_gmaps(item, location_name)

            if parsed is None:
                skipped_empty += 1
                continue

            h = text_hash(parsed["text"])
            if h in seen:
                skipped_dup += 1
                continue
            seen.add(h)

            parsed["text_hash"]    = h
            parsed["collected_at"] = datetime.utcnow().isoformat()
            rows.append(parsed)
            added += 1

        print(f"    → {added} adăugate | {skipped_empty} fără text | {skipped_dup} duplicate")

    df = pd.DataFrame(rows, columns=[
        "source", "location", "text", "rating",
        "is_recommended", "likes",
        "author", "review_date", "url",
        "text_hash", "collected_at",
    ])
    df = df[df["text"].str.len() > 5].reset_index(drop=True)
    df.index.name = "review_id"

    out_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reviews_merged.csv")
    df.to_csv(out_path, index=True, encoding="utf-8")

    print(f"\n✓ Total: {len(df)} recenzii unice → {out_path}")
    print(df["source"].value_counts().to_string())
    if df["rating"].notna().any():
        print(f"\nRating mediu (Google Maps): {df['rating'].mean():.2f} / 5")
        print(df["rating"].value_counts().sort_index().to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directorul cu facebook.json, gmaps_ctp.json, gmaps_ratp.json")
    args = parser.parse_args()
    merge(data_dir=args.data_dir)