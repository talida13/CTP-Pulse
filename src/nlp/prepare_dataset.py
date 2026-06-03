"""
prepare_dataset.py
------------------
Pregătește datasetul ABSA pentru două modele ML:

Model 1: Aspect detector
    input:  aspect + text
    output: present / absent

Model 2: Sentiment per aspect
    input:  aspect + fragment + text
    output: negativ / neutru / pozitiv

Input:
    data/processed/absa_flat_dataset_extended.csv

Output:
    data/processed/prepared/
        reviews_train.csv
        reviews_val.csv
        reviews_test.csv

        aspect_pairs_train.csv
        aspect_pairs_val.csv
        aspect_pairs_test.csv

        sentiment_pairs_train.csv
        sentiment_pairs_val.csv
        sentiment_pairs_test.csv

        metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT = Path("data/processed/absa_flat_dataset_extended.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/prepared")

ASPECTS = [
    "punctualitate",
    "frecventa",
    "aglomeratie",
    "validare_bilete",
    "pret",
    "controlori",
    "soferi_vatmani",
    "confort_termic",
    "curatenie",
    "stare_vehicule",
    "informare_calatori",
    "infrastructura_statii",
    "acoperire_rute",
    "relatii_clienti",
    "organizare",
    "siguranta",
]

SENTIMENTS = ["negativ", "neutru", "pozitiv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    return parser.parse_args()


def normalize_sentiment(value: Any) -> str:
    label = str(value).strip().lower()

    if label in {"negative", "negativ"}:
        return "negativ"
    if label in {"neutral", "neutru"}:
        return "neutru"
    if label in {"positive", "pozitiv"}:
        return "pozitiv"

    raise ValueError(f"Sentiment necunoscut: {value}")


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "review_id",
        "text",
        "aspect",
        "sentiment",
        "fragment",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii: {sorted(missing)}")

    invalid_aspects = sorted(set(df["aspect"].dropna().astype(str)) - set(ASPECTS))
    if invalid_aspects:
        raise ValueError(f"Aspecte necunoscute în dataset: {invalid_aspects}")

    invalid_sentiments = []

    for value in df["sentiment"].dropna().unique():
        try:
            normalize_sentiment(value)
        except ValueError:
            invalid_sentiments.append(value)

    if invalid_sentiments:
        raise ValueError(f"Sentimente necunoscute: {invalid_sentiments}")


def build_reviews(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for review_id, group in df.groupby("review_id", dropna=False):
        first = group.iloc[0]
        present_aspects = sorted(set(group["aspect"].dropna().astype(str)))

        row = {
            "review_id": str(review_id),
            "text": str(first["text"]),
            "source": first.get("source", ""),
            "location": first.get("location", ""),
            "rating": first.get("rating", ""),
            "review_date": first.get("review_date", ""),
            "present_aspects": json.dumps(present_aspects, ensure_ascii=False),
            "aspect_count": len(present_aspects),
        }

        for aspect in ASPECTS:
            row[aspect] = 1 if aspect in present_aspects else 0

        rows.append(row)

    return pd.DataFrame(rows)


def split_reviews(reviews: pd.DataFrame, seed: int, val_size: float, test_size: float):
    """
    Split 80/10/10 aproximativ, pe review_id.
    Încercăm stratificare după numărul de aspecte per review.
    """

    stratify = reviews["aspect_count"].clip(upper=4)

    try:
        train_val, test = train_test_split(
            reviews,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_val, test = train_test_split(
            reviews,
            test_size=test_size,
            random_state=seed,
        )

    relative_val_size = val_size / (1.0 - test_size)
    stratify_train_val = train_val["aspect_count"].clip(upper=4)

    try:
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=seed,
            stratify=stratify_train_val,
        )
    except ValueError:
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=seed,
        )

    return train.copy(), val.copy(), test.copy()


def aspect_input(aspect: str, text: str) -> str:
    return f"aspect: {aspect}. text: {text}"


def sentiment_input(aspect: str, fragment: str, text: str) -> str:
    fragment = str(fragment)
    if not fragment or fragment == "nan":
        fragment = text

    return f"aspect: {aspect}. fragment: {fragment}. text: {text}"


def build_aspect_pairs(
    reviews_split: pd.DataFrame,
    seed: int,
    negatives_per_positive: int,
    mode: str,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for _, row in reviews_split.iterrows():
        review_id = str(row["review_id"])
        text = str(row["text"])
        present_aspects = set(json.loads(row["present_aspects"]))

        positive_aspects = sorted(present_aspects)
        negative_aspects = sorted(set(ASPECTS) - present_aspects)

        for aspect in positive_aspects:
            rows.append(
                {
                    "review_id": review_id,
                    "input": aspect_input(aspect, text),
                    "aspect": aspect,
                    "label": 1,
                }
            )

        if mode == "train":
            k = min(
                len(negative_aspects),
                max(1, negatives_per_positive * max(1, len(positive_aspects))),
            )
            sampled_negatives = rng.sample(negative_aspects, k=k)
        else:
            # Pentru val/test păstrăm toate negativele, ca să vedem false positives realist.
            sampled_negatives = negative_aspects

        for aspect in sampled_negatives:
            rows.append(
                {
                    "review_id": review_id,
                    "input": aspect_input(aspect, text),
                    "aspect": aspect,
                    "label": 0,
                }
            )

    return pd.DataFrame(rows)


def build_sentiment_pairs(flat: pd.DataFrame, reviews_split: pd.DataFrame) -> pd.DataFrame:
    split_ids = set(reviews_split["review_id"].astype(str))
    part = flat[flat["review_id"].astype(str).isin(split_ids)].copy()

    rows = []

    for _, row in part.iterrows():
        text = str(row["text"])
        aspect = str(row["aspect"])
        fragment = str(row.get("fragment", ""))

        try:
            sentiment = normalize_sentiment(row["sentiment"])
        except ValueError:
            continue

        rows.append(
            {
                "review_id": str(row["review_id"]),
                "input": sentiment_input(aspect, fragment, text),
                "aspect": aspect,
                "text": text,
                "fragment": fragment,
                "sentiment": sentiment,
                "label": SENTIMENTS.index(sentiment),
            }
        )

    return pd.DataFrame(rows)


def check_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_ids = set(train["review_id"].astype(str))
    val_ids = set(val["review_id"].astype(str))
    test_ids = set(test["review_id"].astype(str))

    assert train_ids.isdisjoint(val_ids), "Leakage între train și val"
    assert train_ids.isdisjoint(test_ids), "Leakage între train și test"
    assert val_ids.isdisjoint(test_ids), "Leakage între val și test"


def print_distribution(name: str, frame: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("rows:", len(frame))
    print("reviews:", frame["review_id"].astype(str).nunique())

    if "sentiment" in frame.columns:
        print("sentiment:")
        print(frame["sentiment"].value_counts().to_string())

    if "aspect" in frame.columns:
        print("aspect:")
        print(frame["aspect"].value_counts().to_string())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flat = pd.read_csv(args.input)
    validate_input(flat)

    flat["review_id"] = flat["review_id"].astype(str)
    flat["sentiment"] = flat["sentiment"].apply(normalize_sentiment)

    reviews = build_reviews(flat)

    train_reviews, val_reviews, test_reviews = split_reviews(
        reviews,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    check_no_leakage(train_reviews, val_reviews, test_reviews)

    # Reviews split
    train_reviews.to_csv(args.output_dir / "reviews_train.csv", index=False)
    val_reviews.to_csv(args.output_dir / "reviews_val.csv", index=False)
    test_reviews.to_csv(args.output_dir / "reviews_test.csv", index=False)

    # Aspect pairs
    aspect_train = build_aspect_pairs(
        train_reviews,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
        mode="train",
    )

    aspect_val = build_aspect_pairs(
        val_reviews,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
        mode="val",
    )

    aspect_test = build_aspect_pairs(
        test_reviews,
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
        mode="test",
    )

    aspect_train.to_csv(args.output_dir / "aspect_pairs_train.csv", index=False)
    aspect_val.to_csv(args.output_dir / "aspect_pairs_val.csv", index=False)
    aspect_test.to_csv(args.output_dir / "aspect_pairs_test.csv", index=False)

    # Sentiment pairs
    sentiment_train = build_sentiment_pairs(flat, train_reviews)
    sentiment_val = build_sentiment_pairs(flat, val_reviews)
    sentiment_test = build_sentiment_pairs(flat, test_reviews)

    sentiment_train.to_csv(args.output_dir / "sentiment_pairs_train.csv", index=False)
    sentiment_val.to_csv(args.output_dir / "sentiment_pairs_val.csv", index=False)
    sentiment_test.to_csv(args.output_dir / "sentiment_pairs_test.csv", index=False)

    metadata = {
        "input": str(args.input),
        "aspects": ASPECTS,
        "sentiments": SENTIMENTS,
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "negatives_per_positive": args.negatives_per_positive,
    }

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved prepared dataset to:", args.output_dir)

    print_distribution("reviews_train", train_reviews)
    print_distribution("reviews_val", val_reviews)
    print_distribution("reviews_test", test_reviews)

    print_distribution("aspect_pairs_train", aspect_train)
    print_distribution("aspect_pairs_val", aspect_val)
    print_distribution("aspect_pairs_test", aspect_test)

    print_distribution("sentiment_pairs_train", sentiment_train)
    print_distribution("sentiment_pairs_val", sentiment_val)
    print_distribution("sentiment_pairs_test", sentiment_test)


if __name__ == "__main__":
    main()
