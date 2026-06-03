from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
DEFAULT_TRAIN_PATH = Path("data/processed/prepared/aspect_pairs_train.csv")
DEFAULT_VAL_PATH = Path("data/processed/prepared/aspect_pairs_val.csv")
DEFAULT_OUTPUT_DIR = Path("model/ctp_aspect_pair_detector")

DEFAULT_ASPECTS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    return parser.parse_args()


def aspect_prompt(aspect: str, text: str) -> str:
    return f"aspect: {aspect}. text: {text}"


def build_review_frame(flat: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    rows = []

    for review_id, group in flat.groupby("review_id", dropna=False):
        text = str(group.iloc[0]["text"])
        present = sorted(set(group["aspect"].dropna().astype(str)) & set(aspects))

        rows.append(
            {
                "review_id": str(review_id),
                "text": text,
                "present_aspects": present,
            }
        )

    return pd.DataFrame(rows)


def build_pair_frame(
    review_frame: pd.DataFrame,
    aspects: list[str],
    seed: int,
    negatives_per_positive: int,
    mode: str,
) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    aspect_set = set(aspects)

    for _, row in review_frame.iterrows():
        review_id = row["review_id"]
        text = row["text"]
        positives = list(row["present_aspects"])
        negatives = sorted(aspect_set - set(positives))

        for aspect in positives:
            records.append(
                {
                    "review_id": review_id,
                    "input": aspect_prompt(aspect, text),
                    "aspect": aspect,
                    "label": 1,
                }
            )

        if mode == "train":
            k = min(len(negatives), max(1, negatives_per_positive * max(1, len(positives))))
            sampled_negatives = rng.sample(negatives, k=k)
        else:
            # La validare testăm toate aspectele negative, ca să vedem realistic false positives.
            sampled_negatives = negatives

        for aspect in sampled_negatives:
            records.append(
                {
                    "review_id": review_id,
                    "input": aspect_prompt(aspect, text),
                    "aspect": aspect,
                    "label": 0,
                }
            )

    return pd.DataFrame(records)


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]

        enc = self.tokenizer(
            str(row["input"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = int(row["label"])
        return item


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": args.seed,
        "dataloader_pin_memory": False,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_prob = probs[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(labels).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def load_prepared_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"review_id", "input", "aspect", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Lipsesc coloane din {path}: {sorted(missing)}")

    df = df.copy()
    df["review_id"] = df["review_id"].astype(str)
    df["input"] = df["input"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    df["label"] = df["label"].astype(int)

    return df
def main() -> None:
    args = parse_args()

    train_pairs = load_prepared_pairs(args.train_path)
    val_pairs = load_prepared_pairs(args.val_path)

    aspects = sorted(
        set(train_pairs["aspect"].dropna().astype(str))
        | set(val_pairs["aspect"].dropna().astype(str))
    )

    # Păstrăm ordinea din lista fixă
    aspects = [a for a in DEFAULT_ASPECTS if a in aspects]

    print("\nTrain path:")
    print(args.train_path)

    print("\nVal path:")
    print(args.val_path)

    print("\nPerechi aspect-text:")
    print("train:", len(train_pairs), train_pairs["label"].value_counts().to_dict())
    print("val:", len(val_pairs), val_pairs["label"].value_counts().to_dict())

    print("\nAspecte:")
    for aspect in aspects:
        train_count = train_pairs[
            (train_pairs["aspect"] == aspect) & (train_pairs["label"] == 1)
        ]["review_id"].nunique()

        val_count = val_pairs[
            (val_pairs["aspect"] == aspect) & (val_pairs["label"] == 1)
        ]["review_id"].nunique()

        print(f"- {aspect}: train positives={train_count}, val positives={val_count}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "absent", 1: "present"},
        label2id={"absent": 0, "present": 1},
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
    )

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=PairDataset(train_pairs, tokenizer, args.max_length),
        eval_dataset=PairDataset(val_pairs, tokenizer, args.max_length),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model_name": args.base_model,
        "aspects": aspects,
        "model_type": "ctp_aspect_pair_detector",
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
    }

    (args.output_dir / "aspect_pair_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nModel salvat în: {args.output_dir}")


if __name__ == "__main__":
    main()
