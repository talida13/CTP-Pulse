"""
predict.py
----------
Pipeline ABSA pentru un review nou.

Model 1 detecteaza aspectele prezente in text, apoi Model 2 clasifica
sentimentul pentru fiecare aspect detectat. Rezultatul este un dictionar
gata de afisat in Streamlit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification


DEFAULT_ASPECT_MODEL_DIR = Path("model/ctp_aspect_pair_detector")
DEFAULT_SENTIMENT_MODEL_DIR = Path("model/ctp_sentiment_pair")
DEFAULT_ASPECT_THRESHOLD = 0.80
MAX_LENGTH = 256


class PredictionError(RuntimeError):
    """Eroare controlata pentru lipsa modelelor sau configuratie invalida."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    project_candidate = _project_root() / candidate
    if project_candidate.exists():
        return project_candidate

    return candidate


def aspect_input(aspect: str, text: str) -> str:
    return f"aspect: {aspect}. text: {text}"


def sentiment_input(aspect: str, text: str, fragment: str | None = None) -> str:
    fragment = fragment if fragment and str(fragment).strip() and str(fragment) != "nan" else text
    return f"aspect: {aspect}. fragment: {fragment}. text: {text}"


def split_fragments(text: str) -> list[str]:
    fragments = []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    for sentence in sentences:
        parts = re.split(r"\s*(?:,|;| iar | dar | însă | insa | și | si )\s*", sentence)
        fragments.extend(part.strip(" .!?;,:") for part in parts if part.strip(" .!?;,:"))

    return fragments or [text.strip()]


def softmax(logits: Any) -> Any:
    import torch

    return torch.softmax(logits, dim=-1)[0]


def load_aspects(aspect_model_dir: Path) -> list[str]:
    meta_path = aspect_model_dir / "aspect_pair_config.json"
    if not meta_path.exists():
        raise PredictionError(
            f"Lipseste configuratia pentru modelul de aspecte: {meta_path}. "
            "Ruleaza mai intai train_aspects.py sau copiaza modelul antrenat in model/ctp_aspect_pair_detector."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    aspects = meta.get("aspects")
    if not aspects:
        raise PredictionError(f"Configuratia {meta_path} nu contine lista 'aspects'.")

    return [str(aspect) for aspect in aspects]


def get_id2label(model: "AutoModelForSequenceClassification") -> dict[int, str]:
    return {int(key): str(value) for key, value in model.config.id2label.items()}


def disable_transformers_vision_imports() -> None:
    """Evita importul torchvision, care nu este necesar pentru modelele text BERT."""
    try:
        from transformers.utils import import_utils

        import_utils._torchvision_available = False
        import_utils._torchvision_version = None
    except Exception:
        pass


@dataclass
class AspectPrediction:
    aspect: str
    sentiment: str
    score: float
    aspect_score: float
    fragment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "score": self.score,
            "aspect_score": self.aspect_score,
            "fragment": self.fragment,
        }


class ABSAPipeline:
    def __init__(
        self,
        aspect_model_dir: Path | str = DEFAULT_ASPECT_MODEL_DIR,
        sentiment_model_dir: Path | str = DEFAULT_SENTIMENT_MODEL_DIR,
        aspect_threshold: float = DEFAULT_ASPECT_THRESHOLD,
    ) -> None:
        self.aspect_model_dir = _resolve_path(aspect_model_dir)
        self.sentiment_model_dir = _resolve_path(sentiment_model_dir)
        self.aspect_threshold = aspect_threshold

        self._validate_model_dir(self.aspect_model_dir, "aspecte")
        self._validate_model_dir(self.sentiment_model_dir, "sentiment")

        self.aspects = load_aspects(self.aspect_model_dir)

        disable_transformers_vision_imports()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.aspect_tokenizer = AutoTokenizer.from_pretrained(self.aspect_model_dir)
        self.aspect_model = AutoModelForSequenceClassification.from_pretrained(self.aspect_model_dir)
        self.aspect_model.eval()

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_dir)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_dir)
        self.sentiment_model.eval()
        self.sentiment_id2label = get_id2label(self.sentiment_model)

    @staticmethod
    def _validate_model_dir(model_dir: Path, label: str) -> None:
        if not model_dir.exists():
            raise PredictionError(f"Nu exista directorul pentru modelul de {label}: {model_dir}")
        if not (model_dir / "config.json").exists():
            raise PredictionError(f"Directorul {model_dir} nu pare sa contina un model Hugging Face valid.")

    def _predict_aspect_probability(self, aspect: str, text: str) -> float:
        import torch

        encoded = self.aspect_tokenizer(
            aspect_input(aspect, text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = self.aspect_model(**encoded)

        probs = softmax(outputs.logits)
        return float(probs[1])

    def _extract_fragment_for_aspect(self, aspect: str, text: str) -> str:
        fragments = split_fragments(text)
        if len(fragments) == 1:
            return fragments[0]

        scored_fragments = []
        for fragment in fragments:
            model_score = self._predict_aspect_probability(aspect, fragment)
            scored_fragments.append((model_score, -len(fragment), fragment))

        _, _, fragment = max(scored_fragments)
        return fragment

    def _predict_sentiment(self, aspect: str, text: str, fragment: str) -> tuple[str, float]:
        import torch

        encoded = self.sentiment_tokenizer(
            sentiment_input(aspect, text, fragment),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = self.sentiment_model(**encoded)

        probs = softmax(outputs.logits)
        pred_id = int(torch.argmax(probs).item())
        return self.sentiment_id2label[pred_id], float(probs[pred_id])

    @staticmethod
    def _overall_sentiment(predictions: list[AspectPrediction]) -> tuple[str, float]:
        if not predictions:
            return "neutru", 0.0

        scores: dict[str, float] = {}
        for pred in predictions:
            scores[pred.sentiment] = scores.get(pred.sentiment, 0.0) + pred.score

        sentiment, total = max(scores.items(), key=lambda item: item[1])
        confidence = total / sum(scores.values())
        return sentiment, confidence

    def predict(self, text: str) -> dict[str, Any]:
        text = str(text).strip()
        if not text:
            raise PredictionError("Textul review-ului este gol.")

        predictions: list[AspectPrediction] = []

        for aspect in self.aspects:
            aspect_score = self._predict_aspect_probability(aspect, text)
            if aspect_score < self.aspect_threshold:
                continue

            fragment = self._extract_fragment_for_aspect(aspect, text)
            sentiment, sentiment_score = self._predict_sentiment(
                aspect=aspect,
                text=text,
                fragment=fragment,
            )
            predictions.append(
                AspectPrediction(
                    aspect=aspect,
                    sentiment=sentiment,
                    score=sentiment_score,
                    aspect_score=aspect_score,
                    fragment=fragment,
                )
            )

        overall_sentiment, overall_score = self._overall_sentiment(predictions)

        return {
            "text": text,
            "overall_sentiment": overall_sentiment,
            "overall_score": overall_score,
            "aspects": [prediction.to_dict() for prediction in predictions],
        }


def load_pipeline(
    aspect_model_dir: Path | str = DEFAULT_ASPECT_MODEL_DIR,
    sentiment_model_dir: Path | str = DEFAULT_SENTIMENT_MODEL_DIR,
    aspect_threshold: float = DEFAULT_ASPECT_THRESHOLD,
) -> ABSAPipeline:
    return ABSAPipeline(
        aspect_model_dir=aspect_model_dir,
        sentiment_model_dir=sentiment_model_dir,
        aspect_threshold=aspect_threshold,
    )


def predict_review(text: str, pipeline: ABSAPipeline | None = None) -> dict[str, Any]:
    pipeline = pipeline or load_pipeline()
    return pipeline.predict(text)
