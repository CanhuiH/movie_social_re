from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import (
    get_movie_global_category_mapping,
    get_movie_relationship_definitions,
    get_movie_relationship_labels,
)


DEFAULT_RELATIONSHIP_LABEL = "unclear"


@dataclass
class TurnWindow:
    """A single turn-level example used for relationship extraction."""

    movie_name: str
    movie_idx: str | int | None = None
    conversation_id: str = ""
    utterance_id: str = ""
    timestamp: Any = None
    turn_index: int | None = None
    utterance_order: int | None = None
    speaker_id: str = ""
    speaker_name: str | None = None
    listener_id: str | None = None
    listener_name: str | None = None
    reply_to: str | None = None
    current_turn: str = ""
    current_text: str = ""
    context_text: str = ""
    num_prev_turns: int = 0
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
    global_category: str = DEFAULT_RELATIONSHIP_LABEL
    confidence: float = 0.0
    evidence: str = ""
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewedRelationPrediction(RelationPrediction):
    """A relation prediction after optional human review or adjudication."""

    reviewed_relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    reviewer_notes: str = ""

    def final_relationship_type(self) -> str:
        return normalize_relationship_label(
            self.reviewed_relationship_type,
            movie_name=self.movie_name,
        )


@dataclass
class GraphEdge:
    """A lightweight directed edge representing speaker -> listener relation state.

    This class is kept for compatibility with earlier graph-related modules. The
    current simplified project may focus only on relationship extraction.
    """

    source_id: str
    source_name: str | None
    target_id: str
    target_name: str | None
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    global_category: str = DEFAULT_RELATIONSHIP_LABEL
    confidence: float = 0.0
    conversation_id: str | None = None
    utterance_id_last_updated: str | None = None
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SocialSummary:
    """A short structured summary derived from relation state.

    This class is kept for compatibility with earlier translation-related modules.
    """

    speaker_name: str | None
    listener_name: str | None
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    global_category: str = DEFAULT_RELATIONSHIP_LABEL
    guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranslationInput:
    """Prompt-ready translation input combining local context and relation summary.

    This class is kept for compatibility with earlier translation-related modules.
    """

    movie_name: str
    conversation_id: str
    utterance_id: str
    speaker_name: str | None
    listener_name: str | None
    current_turn: str
    context_text: str
    relationship_type: str = DEFAULT_RELATIONSHIP_LABEL
    global_category: str = DEFAULT_RELATIONSHIP_LABEL
    social_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_movie_name(movie_name: str | None) -> str:
    """Normalize movie names for schema lookup."""
    return str(movie_name or "").strip().lower()


def get_allowed_relationship_labels(movie_name: str) -> set[str]:
    """Return the allowed movie-specific relationship labels."""
    return set(get_movie_relationship_labels(movie_name))


def normalize_relationship_label(
    label: str | None,
    movie_name: str | None = None,
    default: str = DEFAULT_RELATIONSHIP_LABEL,
) -> str:
    """Normalize a relationship label against the movie-specific schema.

    If movie_name is provided, the label must appear in that movie's customized
    schema. Invalid or missing labels are normalized to `unclear`.
    """
    if label is None:
        return default

    normalized = str(label).strip().lower()
    if not normalized:
        return default

    if movie_name is None:
        return normalized

    allowed_labels = get_allowed_relationship_labels(movie_name)
    return normalized if normalized in allowed_labels else default


def validate_relationship_label(label: str | None, movie_name: str) -> bool:
    """Return True if a label is part of the movie-specific relationship schema."""
    if label is None:
        return False
    normalized = str(label).strip().lower()
    return normalized in get_allowed_relationship_labels(movie_name)


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


def get_global_category(
    relationship_type: str | None,
    movie_name: str,
    default: str = DEFAULT_RELATIONSHIP_LABEL,
) -> str:
    """Map a movie-specific relationship label to a broad global category."""
    normalized = normalize_relationship_label(
        relationship_type,
        movie_name=movie_name,
        default=default,
    )
    mapping = get_movie_global_category_mapping(movie_name)
    return mapping.get(normalized, default)


def relationship_guidance(label: str | None, movie_name: str | None = None) -> str:
    """Map a relationship label to short translation-oriented guidance.

    For customized schemas, the label is first mapped to a broad global category.
    The returned guidance is intentionally general so it can work across movies.
    """
    if movie_name is not None:
        category = get_global_category(label, movie_name=movie_name)
    else:
        category = normalize_relationship_label(label)

    guidance_map = {
        "familial": "Preserve family intimacy, obligation, or family authority in the wording.",
        "romance": "Preserve romantic intimacy, attraction, or emotional closeness when locally supported.",
        "friendship": "Preserve casual, familiar, and friendly wording.",
        "acquaintance": "Use neutral social wording without excessive intimacy or hierarchy.",
        "authority": "Preserve hierarchy, command, rank, discipline, or formal role relations.",
        "class_or_service": "Preserve class, service, patronage, dependency, or status-based deference.",
        "adversarial": "Preserve tension, hostility, coercion, threat, rivalry, or confrontational tone.",
        "ally": "Preserve support, loyalty, protection, cooperation, or shared alignment.",
        "unclear": "Use neutral wording unless local dialogue strongly suggests otherwise.",
    }
    return guidance_map.get(category, guidance_map[DEFAULT_RELATIONSHIP_LABEL])


def get_label_definition(label: str | None, movie_name: str) -> str:
    """Return the movie-specific definition for a relationship label."""
    normalized = normalize_relationship_label(label, movie_name=movie_name)
    definitions = get_movie_relationship_definitions(movie_name)
    return definitions.get(normalized, "")


def build_relation_prediction(
    *,
    movie_name: str,
    conversation_id: str,
    utterance_id: str,
    speaker_id: str,
    speaker_name: str | None,
    listener_id: str | None,
    listener_name: str | None,
    relationship_type: str | None,
    confidence: Any = 0.0,
    evidence: str | None = "",
    raw_response: str | None = None,
) -> RelationPrediction:
    """Build a normalized RelationPrediction object from raw model fields."""
    normalized_relationship = normalize_relationship_label(
        relationship_type,
        movie_name=movie_name,
    )
    global_category = get_global_category(
        normalized_relationship,
        movie_name=movie_name,
    )

    return RelationPrediction(
        movie_name=movie_name,
        conversation_id=str(conversation_id),
        utterance_id=str(utterance_id),
        speaker_id=str(speaker_id),
        speaker_name=speaker_name,
        listener_id=listener_id,
        listener_name=listener_name,
        relationship_type=normalized_relationship,
        global_category=global_category,
        confidence=coerce_confidence(confidence),
        evidence=str(evidence or ""),
        raw_response=raw_response,
    )