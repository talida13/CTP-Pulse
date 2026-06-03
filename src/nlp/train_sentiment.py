
from __future__ import annotations

import argparse
from html import parser
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
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
DEFAULT_TRAIN_PATH = Path("data/processed/prepared/sentiment_pairs_train.csv")
DEFAULT_VAL_PATH = Path("data/processed/prepared/sentiment_pairs_val.csv")
DEFAULT_OUTPUT_DIR = Path("model/ctp_sentiment_pair")

LABELS = ["negativ", "neutru", "pozitiv"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


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
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()

    if label in {"negative", "negativ"}:
        return "negativ"
    if label in {"neutral", "neutru"}:
        return "neutru"
    if label in {"positive", "pozitiv"}:
        return "pozitiv"

    raise ValueError(f"Label sentiment necunoscut: {value}")


def make_input(aspect: str, text: str, fragment: str) -> str:
    # Fragmentul e foarte util fiindcă sentimentul poate fi local, nu pentru tot review-ul.
    fragment = fragment if fragment and fragment != "nan" else text
    return f"aspect: {aspect}. fragment: {fragment}. text: {text}"


def build_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    required = {"text", "aspect", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane: {sorted(missing)}")

    if "fragment" not in df.columns:
        df["fragment"] = ""

    rows = []

    for _, row in df.iterrows():
        text = str(row["text"])
        aspect = str(row["aspect"])
        fragment = str(row.get("fragment", ""))

        try:
            label = normalize_label(row["sentiment"])
        except ValueError:
            continue

        rows.append(
            {
                "input": make_input(aspect, text, fragment),
                "aspect": aspect,
                "text": text,
                "fragment": fragment,
                "sentiment": label,
                "label": LABEL_TO_ID[label],
            }
        )

    return pd.DataFrame(rows)

def load_prepared_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"input", "aspect", "text", "fragment", "sentiment", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane din {path}: {sorted(missing)}")

    df = df.copy()
    df["sentiment"] = df["sentiment"].apply(normalize_label)
    df["label"] = df["sentiment"].map(LABEL_TO_ID).astype(int)

    return df

class SentimentPairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]

        encoded = self.tokenizer(
            str(row["input"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = int(row["label"])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = None
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)

        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


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
        "metric_for_best_model": "f1_macro",
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


def compute_class_weights(train_frame: pd.DataFrame) -> torch.Tensor:
    counts = train_frame["label"].value_counts().to_dict()
    total = len(train_frame)
    num_classes = len(LABELS)

    weights = []

    for idx in range(num_classes):
        count = counts.get(idx, 0)
        if count == 0:
            weight = 1.0
        else:
            weight = total / (num_classes * count)

        # Cap ca să nu explodeze clasa neutru dacă e foarte rară.
        weight = max(0.5, min(weight, 6.0))
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    y_true = np.asarray(labels).astype(int)
    y_pred = np.asarray(logits).argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def main() -> None:
    args = parse_args()

    train_frame = load_prepared_frame(args.train_path)
    val_frame = load_prepared_frame(args.val_path)

    print("\nTrain path:")
    print(args.train_path)

    print("\nVal path:")
    print(args.val_path)

    print("\nDistribuție sentiment train + val:")
    full_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    print(full_frame["sentiment"].value_counts().to_string())

    print("\nTrain sentiment:")
    print(train_frame["sentiment"].value_counts().to_string())

    print("\nVal sentiment:")
    print(val_frame["sentiment"].value_counts().to_string())

    class_weights = compute_class_weights(train_frame)

    print("\nClass weights:")
    for idx, weight in enumerate(class_weights.tolist()):
        print(f"- {ID_TO_LABEL[idx]}: {weight:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        config=config,
    )

    trainer = WeightedTrainer(
        model=model,
        args=make_training_args(args),
        train_dataset=SentimentPairDataset(train_frame, tokenizer, args.max_length),
        eval_dataset=SentimentPairDataset(val_frame, tokenizer, args.max_length),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model_name": args.base_model,
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "model_type": "ctp_sentiment_pair_classifier",
    }

    (args.output_dir / "sentiment_pair_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nModel salvat în: {args.output_dir}")


if __name__ == "__main__":
    main()
