"""
train.py
--------
Fine-tune a Romanian BERT model for ABSA over the fixed CTP aspects.

Input:
    data/processed/dataset_train.csv
    data/processed/dataset_val.csv

Output:
    model/ctp_absa_bert/

Example:
    python -m src.nlp.train --epochs 5 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_train.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data" / "processed" / "dataset_val.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "ctp_absa_bert"
DEFAULT_BASE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"

LABELS = ["negativ", "neutru", "pozitiv", "nementionat"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def normalize_label(value: Any) -> str:
    """Normalize sentiment labels, including diacritic and spelling variants."""
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ț", "t").replace("ţ", "t")

    aliases = {
        "positive": "pozitiv",
        "pos": "pozitiv",
        "negative": "negativ",
        "neg": "negativ",
        "neutral": "neutru",
        "neu": "neutru",
        "nemetionat": "nementionat",
        "nementionat": "nementionat",
        "nemenționat": "nementionat",
        "nentionat": "nementionat",
        "none": "nementionat",
        "nan": "nementionat",
        "": "nementionat",
    }
    return aliases.get(text, text)


def discover_label_columns(frame: pd.DataFrame) -> list[str]:
    aspect_columns = [column for column in frame.columns if column.startswith("label_")]
    label_columns = ["overall_sentiment", *aspect_columns]
    missing = [column for column in label_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required label columns: {missing}")
    return label_columns


class ABSADataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label_columns: list[str],
        max_length: int,
    ) -> None:
        if "text" not in frame.columns:
            raise ValueError("Dataset must contain a 'text' column.")

        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.label_columns = label_columns
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [
                [self.encode_label(row[column], column) for column in label_columns]
                for _, row in frame.iterrows()
            ],
            dtype=torch.long,
        )

    @staticmethod
    def encode_label(value: Any, column: str) -> int:
        label = normalize_label(value)
        if label not in LABEL_TO_ID:
            valid = ", ".join(LABELS)
            raise ValueError(f"Unknown label {value!r} in {column}. Expected one of: {valid}")
        return LABEL_TO_ID[label]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


class ABSAConfig(PretrainedConfig):
    model_type = "ctp_absa_bert"

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        head_names: list[str] | None = None,
        num_labels: int = len(LABELS),
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.head_names = head_names or []
        self.num_labels = num_labels
        self.dropout = dropout
        self.id2label = {str(index): label for index, label in ID_TO_LABEL.items()}
        self.label2id = LABEL_TO_ID


class ABSAMultiHeadModel(PreTrainedModel):
    config_class = ABSAConfig

    def __init__(self, config: ABSAConfig) -> None:
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_size, config.num_labels) for _ in config.head_names]
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)
        logits = torch.stack([head(pooled) for head in self.classifiers], dim=1)
        result = {"logits": logits}

        if labels is not None:
            losses = [
                self.loss_fn(logits[:, head_index, :], labels[:, head_index])
                for head_index in range(logits.shape[1])
            ]
            result["loss"] = torch.stack(losses).sum()

        return result


@dataclass
class DataBundle:
    train_dataset: ABSADataset
    val_dataset: ABSADataset
    label_columns: list[str]


def load_data(
    train_csv: Path,
    val_csv: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> DataBundle:
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)
    label_columns = discover_label_columns(train_frame)

    missing_in_val = [column for column in label_columns if column not in val_frame.columns]
    if missing_in_val:
        raise ValueError(f"Validation dataset is missing columns: {missing_in_val}")

    return DataBundle(
        train_dataset=ABSADataset(train_frame, tokenizer, label_columns, max_length),
        val_dataset=ABSADataset(val_frame, tokenizer, label_columns, max_length),
        label_columns=label_columns,
    )


def make_compute_metrics(label_columns: list[str]):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.asarray(predictions).argmax(axis=-1)
        label_ids = np.asarray(labels)

        metrics: dict[str, float] = {}
        per_head_scores = []
        for head_index, column in enumerate(label_columns):
            score = f1_score(
                label_ids[:, head_index],
                pred_ids[:, head_index],
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
            metric_name = column.replace("label_", "f1_")
            metrics[metric_name] = float(score)
            per_head_scores.append(score)

        metrics["macro_f1"] = float(np.mean(per_head_scores))
        return metrics

    return compute_metrics


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    common_args = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    try:
        return TrainingArguments(
            **common_args,
            eval_strategy="epoch",
            save_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            **common_args,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )


def save_metadata(output_dir: Path, label_columns: list[str], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.base_model,
        "max_length": args.max_length,
        "head_names": label_columns,
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
    }
    (output_dir / "absa_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Romanian BERT for CTP ABSA.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    data = load_data(args.train_csv, args.val_csv, tokenizer, args.max_length)

    config = ABSAConfig(
        base_model_name=args.base_model,
        head_names=data.label_columns,
        dropout=args.dropout,
    )
    model = ABSAMultiHeadModel(config)

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=data.train_dataset,
        eval_dataset=data.val_dataset,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(data.label_columns),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            )
        ],
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    save_metadata(args.output_dir, data.label_columns, args)

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
