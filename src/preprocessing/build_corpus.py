"""
build_corpus.py
---------------
Unifică fișierele colectate (Google Maps x2, Facebook) și face
pre-adnotare automată bazată pe dicționarul de aspecte.

Output:
  data/processed/corpus_raw.csv        — toate recenziile unificate
  data/processed/corpus_annotated.csv  — cu pre-adnotare aspect + polaritate
  data/processed/corpus_for_review.csv — format pentru adnotare manuală echipă

Rulare:
  python src/preprocessing/build_corpus.py \
    --gmaps1 data/raw/gmaps_ratp.json \
    --gmaps2 data/raw/gmaps_ctp.json \
    --facebook data/raw/facebook.json

Owner: Caraman Talida + toată echipa (adnotare manuală)
"""

import json
import re
import csv
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

# ── Dicționar aspecte ─────────────────────────────────────────────────────────
# Fiecare aspect are cuvinte cheie și fraze trigger în română.
# Adaugă termeni noi pe măsură ce găsești în corpus.

ASPECT_LEXICON = {
    "punctualitate": {
        "label": "Punctualitate / Întârzieri",
        "keywords": [
            "întârzie", "întârziere", "întârzieri", "târziu", "nu vine", "nu apare",
            "aștept", "așteptare", "timp de așteptare", "nu respectă", "orar",
            "nepunctual", "30 de minute", "40 de minute", "20 de minute", "10 minute",
            "vine rar", "nu mai vine", "aleatoriu", "fara logica", "capăt de linie"
        ],
    },
    "aglomeratie": {
        "label": "Aglomerație",
        "keywords": [
            "aglomerat", "aglomerație", "plin", "înghesuit", "înghesuiți",
            "nu încap", "înghesuială", "ticsit", "supraaglomerat",
            "stau în picioare", "nu ai loc", "tramvaie pline", "pline"
        ],
    },
    "curatenie": {
        "label": "Curățenie vehicul",
        "keywords": [
            "murdar", "mizerie", "miros", "urât miroase", "neîngrijit",
            "gunoi", "praf", "curat", "igienă", "dezinfectat", "jalnic"
        ],
    },
    "sofer": {
        "label": "Comportament șofer / personal",
        "keywords": [
            "șofer", "conducător", "nepoliticos", "agresiv", "înjură",
            "nu așteaptă", "pleacă", "frânează", "brusc", "politicos",
            "controlor", "recuperator", "bun simț", "atitudine", "personal"
        ],
    },
    "aer_conditionat": {
        "label": "Aer condiționat / Căldură",
        "keywords": [
            "cald", "frig", "aer", "climatizare", "aer condiționat",
            "sufocant", "înghețat", "ventilație", "temperatură", "transpiri"
        ],
    },
    "bilet": {
        "label": "Validare bilet / E-ticketing",
        "keywords": [
            "card", "validare", "aparat", "validator", "bilet", "bilete",
            "plată", "abonament", "nu merge", "eroare", "contactless",
            "pos", "compostoare", "compostate", "24pay", "cumpărat bilete",
            "aparatele", "composta", "amenda"
        ],
    },
    "informatii_statie": {
        "label": "Informații stație / Panouri",
        "keywords": [
            "panou", "informații", "orar", "afișaj", "anunț", "display",
            "stație", "nu se vede", "actualizat", "sms", "aplicație"
        ],
    },
    "frecventa": {
        "label": "Frecvența curselor",
        "keywords": [
            "rar", "frecvență", "o dată la", "interval", "prea rar",
            "la 30 de minute", "la 40 de minute", "cursă", "suplimentar",
            "timp de asteptare lung", "asteptare lung"
        ],
    },
    "siguranta_nocturna": {
        "label": "Siguranță nocturnă",
        "keywords": [
            "noaptea", "întunecos", "nesigur", "periculos", "după ora",
            "seara", "stație izolată", "teamă", "incident"
        ],
    },
    "accesibilitate": {
        "label": "Accesibilitate",
        "keywords": [
            "cărucior", "rampă", "accesibil", "dizabilitate",
            "scaun cu rotile", "nevăzător", "handicap", "loc rezervat"
        ],
    },
}

# Cuvinte negative și pozitive pentru polaritate (simplu, fără model ML)
NEGATIVE_WORDS = [
    "incompetență", "incompetent", "jalnic", "îngrozitor", "oribil", "groaznic",
    "eroare", "nu merge", "nu funcționează", "nu funcţionează", "stricat",
    "problema", "problemă", "probleme", "lipsit", "fara", "fără", "nu",
    "nu știi", "aleatoriu", "nu respectă", "atitudine", "recuperator",
    "amenda", "amendă", "0 stele", "1 stea", "cel mai", "cea mai"
]

POSITIVE_WORDS = [
    "bun", "bine", "excelent", "perfect", "mulțumit", "mulțumesc",
    "recomand", "super", "ok", "decent", "acceptabil"
]


# ── Funcții utilitare ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Curățare de bază: lowercase, spații multiple, emoji-uri."""
    if not text:
        return ""
    text = text.lower().strip()
    # Elimină emoji și caractere non-ASCII non-diacritice
    text = re.sub(r'[^\w\s\-.,!?;:\'\"àáâãäăâîșşțţ]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_aspects(text: str) -> list[dict]:
    """
    Detectează aspectele din text prin potrivire de cuvinte cheie.
    Returnează lista de aspecte găsite cu span-ul aproximativ.
    """
    text_lower = text.lower()
    found = []

    for aspect_code, aspect_data in ASPECT_LEXICON.items():
        for kw in aspect_data["keywords"]:
            if kw in text_lower:
                # Găsește poziția în text
                start = text_lower.find(kw)
                found.append({
                    "aspect": aspect_code,
                    "aspect_label": aspect_data["label"],
                    "keyword_matched": kw,
                    "span_start": start,
                    "span_end": start + len(kw),
                })
                break  # un singur match per aspect

    return found


def detect_polarity(text: str, stars: int = None) -> str:
    """
    Estimare polaritate simplă (baseline pre-adnotare).
    Folosește și rating-ul stars dacă e disponibil.
    """
    if stars is not None:
        if stars <= 2:
            return "NEG"
        elif stars >= 4:
            return "POS"
        else:
            return "NEU"

    text_lower = text.lower()
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)

    if neg_count > pos_count:
        return "NEG"
    elif pos_count > neg_count:
        return "POS"
    return "NEU"


def make_id(text: str) -> str:
    """ID unic per recenzie bazat pe conținut."""
    return hashlib.md5(text.encode()).hexdigest()[:10]


# ── Parsare fișiere ───────────────────────────────────────────────────────────

def load_gmaps(filepath: str) -> list[dict]:
    """Încarcă fișier JSON Google Maps (format Apify)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    reviews = []
    for item in data:
        # text = item.get("text", "").strip()
        text = (item.get("text") or "").strip()
        if not text or len(text) < 10:
            continue
        reviews.append({
            "id": make_id(text),
            "source": "google_maps",
            "location": item.get("title", "CTP Iași"),
            "author": item.get("name", "anonim"),
            "stars": item.get("stars"),
            "text_raw": text,
            "text_clean": normalize_text(text),
            "date": item.get("publishedAtDate", ""),
            "url": item.get("reviewUrl", ""),
        })
    return reviews


def load_facebook(filepath: str) -> list[dict]:
    """Încarcă fișier JSON Facebook (format Apify)."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    reviews = []
    for item in data:
        text = (item.get("text") or "").strip()
        if not text or len(text) < 10:
            continue
        # Filtrează postări doar cu emoji (fără conținut textual util)
        clean = normalize_text(text)
        word_count = len(clean.split())
        if word_count < 4:
            continue
        reviews.append({
            "id": make_id(text),
            "source": "facebook",
            "location": "CTP Iași Facebook",
            "author": "anonim",
            "stars": None,
            "text_raw": text,
            "text_clean": clean,
            "date": item.get("date", ""),
            "url": item.get("facebookUrl", ""),
        })
    return reviews


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_corpus(gmaps_files: list[str], facebook_file: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Colectare din toate sursele
    all_reviews = []
    for gf in gmaps_files:
        if Path(gf).exists():
            loaded = load_gmaps(gf)
            print(f"  Google Maps '{gf}': {len(loaded)} recenzii")
            all_reviews.extend(loaded)

    if facebook_file and Path(facebook_file).exists():
        fb = load_facebook(facebook_file)
        print(f"  Facebook: {len(fb)} comentarii")
        all_reviews.extend(fb)

    # 2. Deduplicare pe ID
    seen = set()
    unique = []
    for r in all_reviews:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    print(f"\nTotal unic: {len(unique)} recenzii (din {len(all_reviews)} brute)")

    # 3. Salvare corpus brut
    raw_path = output_path / "corpus_raw.csv"
    fieldnames = ["id", "source", "location", "author", "stars", "text_raw", "text_clean", "date", "url"]
    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique)
    print(f"Salvat: {raw_path}")

    # 4. Pre-adnotare automată
    annotated = []
    no_aspect = []

    for review in unique:
        aspects = detect_aspects(review["text_clean"])
        polarity = detect_polarity(review["text_clean"], review.get("stars"))

        if not aspects:
            # Recenzie fără aspect detectat — merge în coada pentru adnotare manuală
            no_aspect.append(review)
            annotated.append({
                **review,
                "aspects": "OTHER",
                "aspects_labels": "Nedetectat",
                "polarity_auto": polarity,
                "polarity_manual": "",  # de completat manual
                "needs_review": "YES",
                "notes": "",
            })
        else:
            for asp in aspects:
                annotated.append({
                    **review,
                    "aspects": asp["aspect"],
                    "aspects_labels": asp["aspect_label"],
                    "keyword_matched": asp["keyword_matched"],
                    "polarity_auto": polarity,
                    "polarity_manual": "",  # de completat manual
                    "needs_review": "YES" if polarity == "NEU" else "NO",
                    "notes": "",
                })

    # 5. Salvare corpus adnotat
    ann_path = output_path / "corpus_annotated.csv"
    ann_fields = [
        "id", "source", "location", "stars", "text_raw", "text_clean",
        "aspects", "aspects_labels", "keyword_matched",
        "polarity_auto", "polarity_manual", "needs_review", "notes", "date"
    ]
    with open(ann_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ann_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(annotated)
    print(f"Salvat: {ann_path}")

    # 6. Statistici
    print(f"\n── Statistici corpus ──────────────────────────────")
    print(f"Total recenzii unice:     {len(unique)}")
    print(f"Cu aspect detectat:       {len(unique) - len(no_aspect)}")
    print(f"Fără aspect (OTHER):      {len(no_aspect)}")
    print(f"Total rânduri adnotate:   {len(annotated)} (o recenzie poate avea mai multe aspecte)")

    from collections import Counter
    aspect_counts = Counter(r["aspects"] for r in annotated)
    polarity_counts = Counter(r["polarity_auto"] for r in annotated)

    print(f"\nDistribuție aspecte:")
    for asp, count in aspect_counts.most_common():
        print(f"  {asp:<25} {count}")

    print(f"\nDistribuție polaritate (auto):")
    for pol, count in polarity_counts.most_common():
        print(f"  {pol:<10} {count}")

    print(f"\nDe adnotat manual (needs_review=YES): {sum(1 for r in annotated if r['needs_review'] == 'YES')}")
    print(f"\nUrmătorul pas: deschide {ann_path} și completează coloana 'polarity_manual'")
    print("Fiecare membru adnotează ~50 exemple independent → calculați Cohen's Kappa")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ABSA corpus for CTP Pulse")
    parser.add_argument("--gmaps1", default="data/raw/gmaps_ratp.json")
    parser.add_argument("--gmaps2", default="data/raw/gmaps_ctp.json")
    parser.add_argument("--facebook", default="data/raw/facebook.json")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    print("CTP Pulse — build_corpus.py\n")
    build_corpus(
        gmaps_files=[args.gmaps1, args.gmaps2],
        facebook_file=args.facebook,
        output_dir=args.output,
    )
