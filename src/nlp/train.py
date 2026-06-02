"""
train.py
--------
Fine-tuning BERT romanian-cased pentru ABSA pe aspectele CTP.

Input:
    data/processed/dataset_train.csv + dataset_val.csv, daca exista
    sau data/processed/absa_flat_dataset.csv pentru split automat.
Output:
    model/ctp_absa_bert/ (weights + tokenizer + config)
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput


DEFAULT_BASE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("model/ctp_absa_bert")

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

LABELS = ["pozitiv", "negativ", "neutru", "nementionat"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
SENTIMENT_PRIORITY = {
    LABEL_TO_ID["negativ"]: 3,
    LABEL_TO_ID["pozitiv"]: 2,
    LABEL_TO_ID["neutru"]: 1,
    LABEL_TO_ID["nementionat"]: 0,
}

TEXT_COLUMN = "text"
OVERALL_COLUMN = "overall_sentiment"
METADATA_COLUMNS = {
    "review_id",
    "source",
    "location",
    "rating",
    "review_date",
    TEXT_COLUMN,
    OVERALL_COLUMN,
}


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame


class AbsaDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        aspects: list[str],
        max_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        encoded = self.tokenizer(
            str(row[TEXT_COLUMN]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        aspect_labels = [int(row[aspect]) for aspect in self.aspects]
        item["labels"] = torch.tensor(aspect_labels, dtype=torch.long)
        item["overall_labels"] = torch.tensor(int(row[OVERALL_COLUMN]), dtype=torch.long)
        return item


class CtpAbsaBert(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        aspects: list[str],
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_model_name = base_model_name
        self.aspects = aspects
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.bert = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden_size = int(self.config.hidden_size)
        classifier_dropout = getattr(self.config, "classifier_dropout", None) or dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.aspect_heads = nn.ModuleDict(
            {aspect: nn.Linear(hidden_size, num_labels) for aspect in aspects}
        )
        self.overall_head = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        overall_labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**model_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        aspect_logits = torch.stack(
            [self.aspect_heads[aspect](pooled) for aspect in self.aspects],
            dim=1,
        )
        overall_logits = self.overall_head(pooled)

        loss = None
        if labels is not None and overall_labels is not None:
            aspect_loss = sum(
                F.cross_entropy(aspect_logits[:, idx, :], labels[:, idx])
                for idx in range(len(self.aspects))
            )
            overall_loss = F.cross_entropy(overall_logits, overall_labels)
            loss = aspect_loss + overall_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=(aspect_logits, overall_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "pytorch_model.bin")
        metadata = {
            "base_model_name": self.base_model_name,
            "aspects": self.aspects,
            "labels": LABELS,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "num_labels": self.num_labels,
            "model_type": "ctp_absa_bert_multihead",
        }
        (output_path / "ctp_absa_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.config.save_pretrained(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTP Pulse ABSA BERT model.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=None,
        help="Lista de aspecte. Implicit: aspectele finale cunoscute sau cele gasite in CSV.",
    )
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    if pd.isna(value):
        return "nementionat"
    label = str(value).strip().lower()
    if label in {"nemenționat", "nementionat", "ne mentionat", "nemetionat"}:
        return "nementionat"
    return label


def label_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in ID_TO_LABEL:
        return int(value)
    if isinstance(value, float) and value.is_integer() and int(value) in ID_TO_LABEL:
        return int(value)

    label = normalize_label(value)
    if label not in LABEL_TO_ID:
        raise ValueError(f"Label necunoscut: {value!r}. Labeluri acceptate: {LABELS}")
    return LABEL_TO_ID[label]


def infer_aspects(flat_frame: pd.DataFrame, requested_aspects: list[str] | None) -> list[str]:
    if requested_aspects:
        return requested_aspects

    present = set(flat_frame["aspect"].dropna().astype(str))
    default_present = [aspect for aspect in DEFAULT_ASPECTS if aspect in present]
    if default_present:
        return default_present
    return sorted(present)


def flat_to_wide(flat_frame: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    required = {"review_id", TEXT_COLUMN, "aspect", "sentiment", OVERALL_COLUMN}
    missing = required - set(flat_frame.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane din absa_flat_dataset.csv: {sorted(missing)}")

    unknown_aspects = sorted(set(flat_frame["aspect"].dropna().astype(str)) - set(aspects))
    if unknown_aspects:
        print(f"Atentie: aspecte ignorate pentru training: {unknown_aspects}")

    rows: list[dict[str, Any]] = []
    grouped = flat_frame.groupby("review_id", dropna=False)
    for review_id, group in grouped:
        first = group.iloc[0]
        row: dict[str, Any] = {
            "review_id": review_id,
            TEXT_COLUMN: str(first[TEXT_COLUMN]),
            OVERALL_COLUMN: label_id(first[OVERALL_COLUMN]),
        }
        for aspect in aspects:
            row[aspect] = LABEL_TO_ID["nementionat"]

        for _, item in group.iterrows():
            aspect = str(item["aspect"])
            if aspect in aspects:
                current_id = int(row[aspect])
                next_id = label_id(item["sentiment"])
                if SENTIMENT_PRIORITY[next_id] > SENTIMENT_PRIORITY[current_id]:
                    row[aspect] = next_id
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_wide_frame(frame: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    required = {TEXT_COLUMN, OVERALL_COLUMN, *aspects}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane din dataset: {sorted(missing)}")

    normalized = frame.copy()
    normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].fillna("").astype(str)
    normalized[OVERALL_COLUMN] = normalized[OVERALL_COLUMN].map(label_id)
    for aspect in aspects:
        normalized[aspect] = normalized[aspect].map(label_id)
    return normalized


def load_splits(processed_dir: Path, aspects: list[str] | None, seed: int) -> tuple[DatasetSplits, list[str]]:
    train_path = processed_dir / "dataset_train.csv"
    val_path = processed_dir / "dataset_val.csv"
    flat_path = processed_dir / "absa_flat_dataset.csv"

    if train_path.exists() and val_path.exists():
        train_raw = pd.read_csv(train_path)
        if aspects is None:
            aspects = [column for column in train_raw.columns if column not in METADATA_COLUMNS]
        train = normalize_wide_frame(train_raw, aspects)
        val = normalize_wide_frame(pd.read_csv(val_path), aspects)
        return DatasetSplits(train=train, val=val), aspects

    if not flat_path.exists():
        raise FileNotFoundError(
            f"Nu am gasit {train_path}, {val_path} sau fallback-ul {flat_path}."
        )

    flat = pd.read_csv(flat_path)
    aspects = infer_aspects(flat, aspects)
    wide = flat_to_wide(flat, aspects)
    stratify = wide[OVERALL_COLUMN] if wide[OVERALL_COLUMN].nunique() > 1 else None
    try:
        train, val = train_test_split(
            wide,
            test_size=0.1,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train, val = train_test_split(wide, test_size=0.1, random_state=seed)
    return DatasetSplits(train=train, val=val), aspects


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        aspect_logits, overall_logits = predictions
    else:
        aspect_logits, overall_logits = predictions[0], predictions[1]

    if isinstance(labels, tuple):
        aspect_labels, overall_labels = labels
    else:
        aspect_labels, overall_labels = labels[0], labels[1]

    aspect_pred = np.asarray(aspect_logits).argmax(axis=-1)
    overall_pred = np.asarray(overall_logits).argmax(axis=-1)
    aspect_true = np.asarray(aspect_labels)
    overall_true = np.asarray(overall_labels)

    aspect_f1 = f1_score(
        aspect_true.reshape(-1),
        aspect_pred.reshape(-1),
        average="macro",
        zero_division=0,
    )
    overall_f1 = f1_score(overall_true, overall_pred, average="macro", zero_division=0)
    return {
        "f1": float((aspect_f1 + overall_f1) / 2),
        "aspect_f1": float(aspect_f1),
        "overall_f1": float(overall_f1),
    }


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": args.seed,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def train() -> None:
    args = parse_args()
    splits, aspects = load_splits(args.processed_dir, args.aspects, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_dataset = AbsaDataset(splits.train, tokenizer, aspects, args.max_length)
    val_dataset = AbsaDataset(splits.val, tokenizer, aspects, args.max_length)
    model = CtpAbsaBert(args.base_model, aspects, num_labels=len(LABELS))

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model salvat in: {args.output_dir}")
    print(f"Aspecte antrenate ({len(aspects)}): {', '.join(aspects)}")


if __name__ == "__main__":
    train()
