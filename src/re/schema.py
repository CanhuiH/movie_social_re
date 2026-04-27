

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

from src.config import DEFAULT_RELATIONSHIP_LABEL, RELATIONSHIP_LABELS


ALLOWED_RELATIONSHIP_LABELS: Final[set[str]] = set(RELATIONSHIP_LABELS)


@dataclass
class TurnWindow:
    """A single turn-level example used for relationship extraction."""

    movie_name: str
    movie_idx: str | int | None
    conversation_id: str
    utterance_id: str
    utterance_order: int
    timestamp: Any
    speaker_id: str
    speaker_name: str | None
    listener_id: str | None
    listener_name: str | None
    reply_to: str | None
    current_turn: str
    current_text: str
    context_text: str
    num_prev_turns: int
    prev_turns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationPrediction:
    """A normalized relationship extraction prediction for one target turn."""

    movie_name: str
    conversation_id: str
    utterance_id: str
    speaker_id: str
    speaker_name: str | None
    listener_id: str | None
    listener_name: str | None
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    confidence: float = 0.0
    evidence: str = ""
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewedRelationPrediction(RelationPrediction):
    """A relation prediction after human review or adjudication."""

    reviewed_relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    reviewer_notes: str = ""

    def final_relationship_type(self) -> str:
        return normalize_relationship_label(self.reviewed_relationship_type)


@dataclass
class GraphEdge:
    """A lightweight directed graph edge representing speaker->listener relation state."""

    source_id: str
    source_name: str | None
    target_id: str
    target_name: str | None
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    confidence: float = 0.0
    conversation_id: str | None = None
    utterance_id_last_updated: str | None = None
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SocialSummary:
    """A short structured summary derived from graph state for translation prompting."""

    speaker_name: str | None
    listener_name: str | None
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranslationInput:
    """Prompt-ready translation input combining local context and social summary."""

    movie_name: str
    conversation_id: str
    utterance_id: str
    speaker_name: str | None
    listener_name: str | None
    current_turn: str
    context_text: str
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    social_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_relationship_label(label: str | None) -> str:
    """Normalize a relationship label into the fixed project schema."""
    if label is None:
        return DEFAULT_RELATIONSHIP_LABEL

    normalized = str(label).strip().lower()
    if not normalized:
        return DEFAULT_RELATIONSHIP_LABEL

    return normalized if normalized in ALLOWED_RELATIONSHIP_LABELS else DEFAULT_RELATIONSHIP_LABEL


def validate_relationship_label(label: str | None) -> bool:
    """Return True if a label is part of the fixed relationship schema."""
    return normalize_relationship_label(label) != DEFAULT_RELATIONSHIP_LABEL or str(label).strip().lower() == DEFAULT_RELATIONSHIP_LABEL


def coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Convert confidence values to a bounded float in [0, 1]."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def relationship_guidance(label: str) -> str:
    """Map a relationship label to a short translation-oriented guidance string."""
    normalized = normalize_relationship_label(label)

    guidance_map = {
        "familial": "Preserve family-related intimacy or authority in the wording.",
        "romantic": "Preserve intimate or emotionally charged partner language.",
        "friend": "Preserve casual, familiar, and friendly wording.",
        "acquaintance": "Use neutral social wording without excessive intimacy.",
        "authority_institutional": "Preserve institutional hierarchy or formal role relations.",
        "class_service": "Preserve class-based or service-role distinctions in wording.",
        "adversarial": "Preserve tension, hostility, or confrontational tone.",
        "unclear": "Use neutral wording unless local dialogue strongly suggests otherwise.",
    }
    return guidance_map.get(normalized, guidance_map[DEFAULT_RELATIONSHIP_LABEL])