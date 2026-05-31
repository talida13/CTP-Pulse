"""
extract_stop_mentions.py
------------------------
Detectează stațiile CTP menționate în recenzii, folosind lista oficială
de stații din Tranzy Stops API, și creează un CSV cu mențiuni geolocalizate.
Owner: Talida Caraman
Input:
- data/processed/reviews_merged.csv
- data/raw/tranzy_stops.json
Output:
- data/processed/review_stop_mentions.csv
Scop:
- Citește review-urile curățate și deduplicate.
- Citește stațiile oficiale CTP din Tranzy.
- Normalizează textele pentru a ignora diferențele de diacritice/capitalizare.
- Caută numele stațiilor în textul review-urilor.
- Leagă fiecare review de stațiile menționate.
- Păstrează coordonatele reale ale stațiilor pentru heatmap în Streamlit.

TODO:
- [ ] Citire review-uri
- [ ] Citire stații Tranzy
- [ ] Normalizare nume stații și texte review
- [ ] Detectare mențiuni de stații
- [ ] Salvare CSV pentru hartă
"""

import os
import re
import json
import unicodedata
import pandas as pd


REVIEWS_PATH = "data/processed/reviews_merged.csv"
STOPS_PATH = "data/raw/tranzy_stops.json"
OUTPUT_PATH = "data/processed/review_stop_mentions.csv"


def normalize_text(text: str) -> str:
    """
    Transformă textul într-o formă simplificată:
    - lowercase
    - fără diacritice
    - fără punctuație
    - spații normalizate
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Normalizează variante comune
    text = text.replace("ș", "s").replace("ț", "t")
    text = text.replace("ă", "a").replace("â", "a").replace("î", "i")

    # C.U.G. -> cug, Tg. -> tg etc.
    text = text.replace(".", "")

    # Păstrăm doar litere/cifre/spații
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_stop_name(stop_name: str) -> str:
    """
    Curăță numele stației:
    - elimină variante de tip (1), (2), (3)
    - normalizează textul
    """
    stop_name = re.sub(r"\s*\([^)]*\)", "", str(stop_name))
    return normalize_text(stop_name)


def make_aliases(stop_name: str) -> set[str]:
    """
    Creează variante de căutare pentru aceeași stație.
    Exemplu:
    - "Podu Roș (1)" -> "podu ros"
    - "Piața M. Eminescu" -> "piata m eminescu", "piata eminescu"
    - "C.U.G. I" -> "cug i", "cug"
    """
    base = clean_stop_name(stop_name)
    aliases = {base}

    tokens = base.split()

    # Elimină inițiale de tip "m" din "piata m eminescu"
    no_initials = " ".join(t for t in tokens if len(t) > 1)
    if len(no_initials) >= 4:
        aliases.add(no_initials)

    # Elimină numerale romane finale simple: I, II, III
    no_roman_suffix = re.sub(r"\b(i|ii|iii|iv)\b$", "", base).strip()
    if len(no_roman_suffix) >= 4:
        aliases.add(no_roman_suffix)

    # Pentru acronime precum c u g -> cug
    compact = base.replace(" ", "")
    if len(compact) >= 3 and any(ch.isdigit() for ch in compact) is False:
        if base in ["c u g", "c u g i", "c u g ii"]:
            aliases.add("cug")

    return {a for a in aliases if len(a) >= 4}


def load_stops() -> pd.DataFrame:
    with open(STOPS_PATH, encoding="utf-8") as f:
        stops = json.load(f)

    stops_df = pd.DataFrame(stops)

    stops_df = stops_df[
        ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "stop_code"]
    ].copy()

    # Păstrăm doar locații de tip stop/platform.
    # Dacă location_type lipsește sau e 0, este o stație/platformă.
    stops_df = stops_df[
        stops_df["location_type"].fillna(0).astype(int) == 0
    ].copy()

    stops_df["stop_base_name"] = stops_df["stop_name"].apply(
        lambda x: re.sub(r"\s*\([^)]*\)", "", str(x)).strip()
    )
    stops_df["stop_key"] = stops_df["stop_base_name"].apply(normalize_text)

    # Agregăm stațiile cu același nume, deoarece pot exista pe ambele sensuri.
    canonical = (
        stops_df
        .groupby("stop_key", as_index=False)
        .agg(
            stop_name=("stop_base_name", "first"),
            stop_lat=("stop_lat", "mean"),
            stop_lon=("stop_lon", "mean"),
            stop_ids=("stop_id", lambda ids: ",".join(map(str, sorted(ids))))
        )
    )

    canonical["aliases"] = canonical["stop_name"].apply(make_aliases)

    return canonical


def detect_stops_in_review(text: str, stops_df: pd.DataFrame) -> list[dict]:
    norm_text = f" {normalize_text(text)} "
    raw_matches = []

    for _, stop in stops_df.iterrows():
        for alias in stop["aliases"]:
            pattern = rf"\b{re.escape(alias)}\b"

            if re.search(pattern, norm_text):
                raw_matches.append({
                    "stop_key": stop["stop_key"],
                    "stop_name": stop["stop_name"],
                    "stop_lat": stop["stop_lat"],
                    "stop_lon": stop["stop_lon"],
                    "stop_ids": stop["stop_ids"],
                    "matched_alias": alias
                })
                break

    # Evităm duplicate de tip "Dacia" + "Piața Dacia" dacă ambele apar.
    filtered = []
    for match in raw_matches:
        alias = match["matched_alias"]

        is_contained_in_longer_match = any(
            alias != other["matched_alias"]
            and f" {alias} " in f" {other['matched_alias']} "
            for other in raw_matches
        )

        if not is_contained_in_longer_match:
            filtered.append(match)

    # Unicitate pe stop_key
    unique = {}
    for match in filtered:
        unique[match["stop_key"]] = match

    return list(unique.values())


def main():
    reviews = pd.read_csv(REVIEWS_PATH)
    reviews["review_id"] = reviews["review_id"].astype(str)

    stops_df = load_stops()

    rows = []

    for _, review in reviews.iterrows():
        matches = detect_stops_in_review(review["text"], stops_df)

        for match in matches:
            rows.append({
                "review_id": review["review_id"],
                "source": review.get("source"),
                "location": review.get("location"),
                "rating": review.get("rating"),
                "text": review.get("text"),
                "stop_name": match["stop_name"],
                "stop_key": match["stop_key"],
                "stop_ids": match["stop_ids"],
                "matched_alias": match["matched_alias"],
                "lat": match["stop_lat"],
                "lon": match["stop_lon"]
            })

    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✓ Salvate {len(out_df)} mențiuni de stații în {OUTPUT_PATH}")

    if not out_df.empty:
        print("\nTop stații menționate:")
        print(out_df["stop_name"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()