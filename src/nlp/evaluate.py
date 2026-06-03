"""
evaluate.py
-----------
Calculează metricile modelului pe setul de test și compară metodele.
Owner: 
Input:  data/processed/dataset_test.csv + model/ctp_absa_bert/
Output: afișare tabel în terminal + data/processed/evaluation_results.csv
TODO:
- [ ] Rulare predicții pe tot dataset_test.csv
- [ ] Calcul Precision, Recall, F1 per aspect pentru modelul BERT
- [ ] Același calcul pentru baseline SentiWordNet (pentru comparație)
- [ ] Tabel comparativ: SentiWordNet vs BERT fine-tuned, per aspect și overall
- [ ] Matrice de confuzie per aspect
- [ ] Salvare rezultate în evaluation_results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_PREPARED_DIR = Path("data/processed/prepared")
DEFAULT_ASPECT_MODEL_DIR = Path("model/ctp_aspect_pair_detector")
DEFAULT_SENTIMENT_MODEL_DIR = Path("model/ctp_sentiment_pair")
DEFAULT_OUTPUT = Path("data/processed/evaluation_results.csv")

ASPECT_THRESHOLD = 0.80

ASPECT_KEYWORDS = {
    "punctualitate": ["întârziat", "intarziat", "la timp", "orar", "program"],
    "frecventa": ["aștept", "astept", "30 de minute", "40 de minute", "rar", "interval", "vine greu"],
    "aglomeratie": ["aglomerat", "aglomerată", "aglomerata", "plin", "înghesuit", "inghesuit", "loc pe scaun"],
    "validare_bilete": ["pos", "bilet", "bilete", "24pay", "compostor", "validare", "card", "automat", "plăti", "plati"],
    "pret": ["preț", "pret", "scump", "tarif", "abonament"],
    "controlori": ["controlor", "controlori", "amendă", "amenda", "control"],
    "soferi_vatmani": ["șofer", "sofer", "vatman", "conducător", "conducator"],
    "confort_termic": ["frig", "cald", "căldură", "caldura", "aer condiționat", "aer conditionat"],
    "curatenie": ["curat", "curată", "curata", "murdar", "mizerie", "mirosea", "miros"],
    "stare_vehicule": ["vechi", "defect", "vehicul", "scaune", "uși", "usi", "funcțional", "functional"],
    "informare_calatori": ["panou", "aplicația", "aplicatia", "afișat", "afisat", "timp de sosire", "informații", "informatii"],
    "infrastructura_statii": ["stație", "statie", "stația", "statia", "refugiu", "adăpost", "adapost", "copertină", "copertina"],
    "acoperire_rute": ["rută", "ruta", "traseu", "linie", "cartier", "zonă", "zona", "legături", "legaturi"],
    "relatii_clienti": ["sesizare", "reclamație", "reclamatie", "call center", "răspuns", "raspuns", "client"],
    "organizare": ["organizare", "program", "management", "companie", "haos", "măsuri", "masuri"],
    "siguranta": ["siguranță", "siguranta", "agresiv", "prudent", "incident", "accident", "pericol", "frână", "frana"],
}

POSITIVE_WORDS = [
    "la timp",
    "corect",
    "rapid",
    "curat",
    "curată",
    "curata",
    "civilizat",
    "civilizați",
    "civilizati",
    "politicos",
    "prudent",
    "siguranță",
    "siguranta",
    "funcționat",
    "functionat",
    "fără probleme",
    "fara probleme",
    "suficient spațiu",
    "suficient spatiu",
]

NEGATIVE_WORDS = [
    "nu merge",
    "nu funcționează",
    "nu functioneaza",
    "nu am putut",
    "întârziat",
    "intarziat",
    "aglomerat",
    "plin",
    "murdar",
    "mizerie",
    "urât",
    "urat",
    "greșit",
    "gresit",
    "agresiv",
    "scump",
    "defect",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--aspect-model-dir", type=Path, default=DEFAULT_ASPECT_MODEL_DIR)
    parser.add_argument("--sentiment-model-dir", type=Path, default=DEFAULT_SENTIMENT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aspect-threshold", type=float, default=ASPECT_THRESHOLD)
    return parser.parse_args()


def softmax_positive(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[1])


def get_id2label(model: AutoModelForSequenceClassification) -> dict[int, str]:
    out = {}
    for k, v in model.config.id2label.items():
        out[int(k)] = v
    return out


def aspect_input(aspect: str, text: str) -> str:
    return f"aspect: {aspect}. text: {text}"


def sentiment_input(aspect: str, text: str, fragment: str | None = None) -> str:
    if fragment is None or not str(fragment).strip() or str(fragment) == "nan":
        fragment = text
    return f"aspect: {aspect}. fragment: {fragment}. text: {text}"


def load_aspects(aspect_model_dir: Path) -> list[str]:
    meta_path = aspect_model_dir / "aspect_pair_config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta["aspects"]


def keyword_aspect_baseline(text: str, aspects: list[str]) -> set[str]:
    low = text.lower()
    detected = set()

    for aspect in aspects:
        kws = ASPECT_KEYWORDS.get(aspect, [])
        if any(kw.lower() in low for kw in kws):
            detected.add(aspect)

    return detected


def lexicon_sentiment_baseline(text: str) -> str:
    low = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in low)
    neg = sum(1 for w in NEGATIVE_WORDS if w in low)

    if neg > pos:
        return "negativ"
    if pos > neg:
        return "pozitiv"
    return "neutru"


def predict_aspect_model(
    text: str,
    aspects: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    threshold: float,
) -> set[str]:
    detected = set()

    for aspect in aspects:
        encoded = tokenizer(
            aspect_input(aspect, text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**encoded)

        prob_present = softmax_positive(outputs.logits)

        if prob_present >= threshold:
            detected.add(aspect)

    return detected


def predict_sentiment_model(
    aspect: str,
    text: str,
    fragment: str | None,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
) -> str:
    encoded = tokenizer(
        sentiment_input(aspect, text, fragment),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**encoded)

    pred_id = int(torch.argmax(outputs.logits, dim=-1)[0])
    id2label = get_id2label(model)

    return id2label[pred_id]


def prf_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate_aspect_detector(
    prepared_dir: Path,
    aspect_model_dir: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    test_path = prepared_dir / "aspect_pairs_test.csv"
    df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(aspect_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(aspect_model_dir)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for _, row in df.iterrows():
        encoded = tokenizer(
            str(row["input"]),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**encoded)

        prob = softmax_positive(outputs.logits)
        pred = int(prob >= threshold)

        y_true.append(int(row["label"]))
        y_pred.append(pred)
        y_prob.append(prob)

    df = df.copy()
    df["pred"] = y_pred
    df["prob_present"] = y_prob

    rows = []

    rows.append({
        "module": "aspect_detector",
        "method": "BERT_pair",
        "scope": "overall",
        "metric": "precision",
        "value": precision_score(y_true, y_pred, zero_division=0),
    })
    rows.append({
        "module": "aspect_detector",
        "method": "BERT_pair",
        "scope": "overall",
        "metric": "recall",
        "value": recall_score(y_true, y_pred, zero_division=0),
    })
    rows.append({
        "module": "aspect_detector",
        "method": "BERT_pair",
        "scope": "overall",
        "metric": "f1",
        "value": f1_score(y_true, y_pred, zero_division=0),
    })
    rows.append({
        "module": "aspect_detector",
        "method": "BERT_pair",
        "scope": "overall",
        "metric": "accuracy",
        "value": accuracy_score(y_true, y_pred),
    })

    for aspect, part in df.groupby("aspect"):
        yt = part["label"].astype(int).tolist()
        yp = part["pred"].astype(int).tolist()

        rows.extend([
            {
                "module": "aspect_detector",
                "method": "BERT_pair",
                "scope": aspect,
                "metric": "precision",
                "value": precision_score(yt, yp, zero_division=0),
            },
            {
                "module": "aspect_detector",
                "method": "BERT_pair",
                "scope": aspect,
                "metric": "recall",
                "value": recall_score(yt, yp, zero_division=0),
            },
            {
                "module": "aspect_detector",
                "method": "BERT_pair",
                "scope": aspect,
                "metric": "f1",
                "value": f1_score(yt, yp, zero_division=0),
            },
        ])

    return rows, df


def evaluate_sentiment_classifier(
    prepared_dir: Path,
    sentiment_model_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    test_path = prepared_dir / "sentiment_pairs_test.csv"
    df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_dir)
    model.eval()

    id2label = get_id2label(model)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        pred_label = predict_sentiment_model(
            aspect=str(row["aspect"]),
            text=str(row["text"]),
            fragment=str(row.get("fragment", "")),
            tokenizer=tokenizer,
            model=model,
        )

        y_true.append(str(row["sentiment"]))
        y_pred.append(pred_label)

    df = df.copy()
    df["pred_sentiment"] = y_pred

    rows = []

    for avg in ["macro", "weighted"]:
        rows.append({
            "module": "sentiment_classifier",
            "method": "BERT_pair",
            "scope": "overall",
            "metric": f"f1_{avg}",
            "value": f1_score(y_true, y_pred, average=avg, zero_division=0),
        })

    rows.extend([
        {
            "module": "sentiment_classifier",
            "method": "BERT_pair",
            "scope": "overall",
            "metric": "accuracy",
            "value": accuracy_score(y_true, y_pred),
        },
        {
            "module": "sentiment_classifier",
            "method": "BERT_pair",
            "scope": "overall",
            "metric": "precision_macro",
            "value": precision_score(y_true, y_pred, average="macro", zero_division=0),
        },
        {
            "module": "sentiment_classifier",
            "method": "BERT_pair",
            "scope": "overall",
            "metric": "recall_macro",
            "value": recall_score(y_true, y_pred, average="macro", zero_division=0),
        },
    ])

    for aspect, part in df.groupby("aspect"):
        yt = part["sentiment"].astype(str).tolist()
        yp = part["pred_sentiment"].astype(str).tolist()

        rows.append({
            "module": "sentiment_classifier",
            "method": "BERT_pair",
            "scope": aspect,
            "metric": "f1_macro",
            "value": f1_score(yt, yp, average="macro", zero_division=0),
        })

    return rows, df


def evaluate_pipeline(
    prepared_dir: Path,
    aspect_model_dir: Path,
    sentiment_model_dir: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    reviews = pd.read_csv(prepared_dir / "reviews_test.csv")
    gold_sent = pd.read_csv(prepared_dir / "sentiment_pairs_test.csv")

    aspects = load_aspects(aspect_model_dir)

    aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_dir)
    aspect_model = AutoModelForSequenceClassification.from_pretrained(aspect_model_dir)
    aspect_model.eval()

    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_dir)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_dir)
    sentiment_model.eval()

    gold_by_review: dict[str, set[tuple[str, str]]] = {}
    gold_aspects_by_review: dict[str, set[str]] = {}

    for _, row in gold_sent.iterrows():
        rid = str(row["review_id"])
        aspect = str(row["aspect"])
        sent = str(row["sentiment"])

        gold_by_review.setdefault(rid, set()).add((aspect, sent))
        gold_aspects_by_review.setdefault(rid, set()).add(aspect)

    detail_rows = []

    aspect_tp = aspect_fp = aspect_fn = 0
    tuple_tp = tuple_fp = tuple_fn = 0

    baseline_aspect_tp = baseline_aspect_fp = baseline_aspect_fn = 0
    baseline_tuple_tp = baseline_tuple_fp = baseline_tuple_fn = 0

    for _, row in reviews.iterrows():
        rid = str(row["review_id"])
        text = str(row["text"])

        gold_tuples = gold_by_review.get(rid, set())
        gold_aspects = gold_aspects_by_review.get(rid, set())

        # BERT pipeline
        pred_aspects = predict_aspect_model(
            text=text,
            aspects=aspects,
            tokenizer=aspect_tokenizer,
            model=aspect_model,
            threshold=threshold,
        )

        pred_tuples = set()

        for aspect in pred_aspects:
            # Realistic ML-only: nu avem fragment la inference, deci folosim textul ca fragment.
            pred_sent = predict_sentiment_model(
                aspect=aspect,
                text=text,
                fragment=text,
                tokenizer=sentiment_tokenizer,
                model=sentiment_model,
            )
            pred_tuples.add((aspect, pred_sent))

        a_tp = len(pred_aspects & gold_aspects)
        a_fp = len(pred_aspects - gold_aspects)
        a_fn = len(gold_aspects - pred_aspects)

        t_tp = len(pred_tuples & gold_tuples)
        t_fp = len(pred_tuples - gold_tuples)
        t_fn = len(gold_tuples - pred_tuples)

        aspect_tp += a_tp
        aspect_fp += a_fp
        aspect_fn += a_fn

        tuple_tp += t_tp
        tuple_fp += t_fp
        tuple_fn += t_fn

        # Baseline pipeline
        base_aspects = keyword_aspect_baseline(text, aspects)
        base_tuples = set()

        for aspect in base_aspects:
            base_sent = lexicon_sentiment_baseline(text)
            base_tuples.add((aspect, base_sent))

        ba_tp = len(base_aspects & gold_aspects)
        ba_fp = len(base_aspects - gold_aspects)
        ba_fn = len(gold_aspects - base_aspects)

        bt_tp = len(base_tuples & gold_tuples)
        bt_fp = len(base_tuples - gold_tuples)
        bt_fn = len(gold_tuples - base_tuples)

        baseline_aspect_tp += ba_tp
        baseline_aspect_fp += ba_fp
        baseline_aspect_fn += ba_fn

        baseline_tuple_tp += bt_tp
        baseline_tuple_fp += bt_fp
        baseline_tuple_fn += bt_fn

        detail_rows.append({
            "review_id": rid,
            "text": text,
            "gold_aspects": sorted(gold_aspects),
            "pred_aspects_bert": sorted(pred_aspects),
            "gold_tuples": sorted(gold_tuples),
            "pred_tuples_bert": sorted(pred_tuples),
            "pred_aspects_baseline": sorted(base_aspects),
            "pred_tuples_baseline": sorted(base_tuples),
        })

    rows = []

    aspect_p, aspect_r, aspect_f1 = prf_from_counts(aspect_tp, aspect_fp, aspect_fn)
    tuple_p, tuple_r, tuple_f1 = prf_from_counts(tuple_tp, tuple_fp, tuple_fn)

    base_aspect_p, base_aspect_r, base_aspect_f1 = prf_from_counts(
        baseline_aspect_tp,
        baseline_aspect_fp,
        baseline_aspect_fn,
    )
    base_tuple_p, base_tuple_r, base_tuple_f1 = prf_from_counts(
        baseline_tuple_tp,
        baseline_tuple_fp,
        baseline_tuple_fn,
    )

    for metric, value in [
        ("aspect_precision", aspect_p),
        ("aspect_recall", aspect_r),
        ("aspect_f1", aspect_f1),
        ("tuple_precision", tuple_p),
        ("tuple_recall", tuple_r),
        ("tuple_f1", tuple_f1),
    ]:
        rows.append({
            "module": "pipeline",
            "method": "BERT_pair_pipeline",
            "scope": "end_to_end",
            "metric": metric,
            "value": value,
        })

    for metric, value in [
        ("aspect_precision", base_aspect_p),
        ("aspect_recall", base_aspect_r),
        ("aspect_f1", base_aspect_f1),
        ("tuple_precision", base_tuple_p),
        ("tuple_recall", base_tuple_r),
        ("tuple_f1", base_tuple_f1),
    ]:
        rows.append({
            "module": "pipeline",
            "method": "keyword_lexicon_baseline",
            "scope": "end_to_end",
            "metric": metric,
            "value": value,
        })

    details = pd.DataFrame(detail_rows)

    return rows, details


def main() -> None:
    args = parse_args()

    all_rows = []

    print("\nEvaluating aspect detector...")
    aspect_rows, aspect_predictions = evaluate_aspect_detector(
        prepared_dir=args.prepared_dir,
        aspect_model_dir=args.aspect_model_dir,
        threshold=args.aspect_threshold,
    )
    all_rows.extend(aspect_rows)

    print("\nEvaluating sentiment classifier...")
    sentiment_rows, sentiment_predictions = evaluate_sentiment_classifier(
        prepared_dir=args.prepared_dir,
        sentiment_model_dir=args.sentiment_model_dir,
    )
    all_rows.extend(sentiment_rows)

    print("\nEvaluating end-to-end pipeline...")
    pipeline_rows, pipeline_details = evaluate_pipeline(
        prepared_dir=args.prepared_dir,
        aspect_model_dir=args.aspect_model_dir,
        sentiment_model_dir=args.sentiment_model_dir,
        threshold=args.aspect_threshold,
    )
    all_rows.extend(pipeline_rows)

    results = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    aspect_predictions.to_csv(args.output.parent / "aspect_test_predictions.csv", index=False)
    sentiment_predictions.to_csv(args.output.parent / "sentiment_test_predictions.csv", index=False)
    pipeline_details.to_csv(args.output.parent / "pipeline_test_predictions.csv", index=False)

    print("\n================ EVALUATION SUMMARY ================")

    summary = results[
        (results["scope"].isin(["overall", "end_to_end"]))
    ].copy()

    print(summary.pivot_table(
        index=["module", "method", "scope"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).round(4).to_string())

    print("\nSaved:")
    print("-", args.output)
    print("-", args.output.parent / "aspect_test_predictions.csv")
    print("-", args.output.parent / "sentiment_test_predictions.csv")
    print("-", args.output.parent / "pipeline_test_predictions.csv")

    print("\n================ ERROR PROPAGATION ================")

    def get_metric(method: str, metric: str) -> float:
        part = results[
            (results["module"] == "pipeline")
            & (results["method"] == method)
            & (results["metric"] == metric)
        ]
        if part.empty:
            return 0.0
        return float(part.iloc[0]["value"])

    aspect_recall = get_metric("BERT_pair_pipeline", "aspect_recall")
    tuple_recall = get_metric("BERT_pair_pipeline", "tuple_recall")

    sentiment_acc_rows = results[
        (results["module"] == "sentiment_classifier")
        & (results["method"] == "BERT_pair")
        & (results["metric"] == "accuracy")
    ]

    sentiment_accuracy = float(sentiment_acc_rows.iloc[0]["value"]) if not sentiment_acc_rows.empty else 0.0

    expected_upper = aspect_recall * sentiment_accuracy

    print(f"Aspect recall Model 1: {aspect_recall:.4f}")
    print(f"Sentiment accuracy Model 2 pe gold aspects: {sentiment_accuracy:.4f}")
    print(f"Estimare simplă impact cumulat: {aspect_recall:.4f} × {sentiment_accuracy:.4f} = {expected_upper:.4f}")
    print(f"Tuple recall end-to-end măsurat: {tuple_recall:.4f}")
    print("Diferența dintre estimare și pipeline arată efecte de propagare + mismatch între training și inference.")


if __name__ == "__main__":
    main()
