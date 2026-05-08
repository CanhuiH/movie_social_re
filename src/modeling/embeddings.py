

"""BERT embedding utilities for power/respect modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.paths import get_embedding_file_path, load_modeling_settings


def resolve_device(device_setting: str = "auto") -> torch.device:
    """Resolve the configured device string to a torch.device."""
    device_setting = str(device_setting).strip().lower()

    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if device_setting == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")

    return torch.device(device_setting)


def mean_pool_hidden_states(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def pool_embeddings(
    model_outputs: Any,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Pool transformer outputs into one embedding per row."""
    pooling = str(pooling).strip().lower()
    last_hidden_state = model_outputs.last_hidden_state

    if pooling == "mean":
        return mean_pool_hidden_states(last_hidden_state, attention_mask)

    if pooling in {"cls", "[cls]"}:
        return last_hidden_state[:, 0, :]

    raise ValueError(f"Unsupported pooling method: {pooling}. Use 'mean' or 'cls'.")


def load_embedding_components(
    model_name: str,
    device: torch.device,
) -> tuple[Any, Any]:
    """Load tokenizer and BERT-style encoder model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_texts_with_bert(
    texts: list[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 16,
    pooling: str = "mean",
    device: torch.device | str = "auto",
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts into BERT embeddings.

    The model is used as a frozen feature extractor. No fine-tuning is performed.
    """
    if not texts:
        raise ValueError("Cannot encode an empty text list.")

    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    tokenizer, model = load_embedding_components(model_name, resolved_device)

    clean_texts = ["" if pd.isna(text) else str(text) for text in texts]
    embeddings: list[np.ndarray] = []

    iterator = range(0, len(clean_texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing BERT embeddings")

    with torch.no_grad():
        for start_idx in iterator:
            batch_texts = clean_texts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}

            outputs = model(**encoded)
            pooled = pool_embeddings(outputs, encoded["attention_mask"], pooling=pooling)
            embeddings.append(pooled.detach().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


def load_embedding_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load embedding settings from modeling_config.json."""
    modeling_settings = settings or load_modeling_settings()
    embedding_settings = modeling_settings.get("embedding", {})
    if not isinstance(embedding_settings, dict):
        raise ValueError("modeling_config.json key 'embedding' must be an object.")
    return embedding_settings


def get_text_column(settings: dict[str, Any] | None = None) -> str:
    """Return the configured combined model text column."""
    modeling_settings = settings or load_modeling_settings()
    return str(modeling_settings.get("combined_text_column", "model_text"))


def compute_embeddings_for_dataframe(
    df: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    text_column: str | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute BERT embeddings for a prepared modeling dataframe."""
    modeling_settings = settings or load_modeling_settings()
    embedding_settings = load_embedding_settings(modeling_settings)
    text_column = text_column or get_text_column(modeling_settings)

    if text_column not in df.columns:
        raise ValueError(f"Text column not found in dataframe: {text_column}")

    texts = df[text_column].fillna("").astype(str).tolist()
    return encode_texts_with_bert(
        texts=texts,
        model_name=str(embedding_settings.get("model_name", "bert-base-uncased")),
        max_length=int(embedding_settings.get("max_length", 256)),
        batch_size=int(embedding_settings.get("batch_size", 16)),
        pooling=str(embedding_settings.get("pooling", "mean")),
        device=str(embedding_settings.get("device", "auto")),
        show_progress=show_progress,
    )


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    """Save embeddings to a .npy file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from a .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return np.load(path)


def get_or_compute_embeddings(
    df: pd.DataFrame,
    split_name: str,
    settings: dict[str, Any] | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Load cached embeddings for a split, or compute and cache them."""
    modeling_settings = settings or load_modeling_settings()
    embedding_settings = load_embedding_settings(modeling_settings)
    cache_embeddings = bool(embedding_settings.get("cache_embeddings", True))
    embedding_path = get_embedding_file_path(split_name)

    if cache_embeddings and embedding_path.exists() and not overwrite:
        embeddings = load_embeddings(embedding_path)
        if len(embeddings) != len(df):
            raise ValueError(
                f"Cached embeddings row count mismatch for split '{split_name}': "
                f"{len(embeddings)} embeddings vs {len(df)} dataframe rows. "
                "Use overwrite=True or pass --overwrite to recompute."
            )
        return embeddings

    embeddings = compute_embeddings_for_dataframe(
        df,
        settings=modeling_settings,
        show_progress=show_progress,
    )

    if cache_embeddings:
        save_embeddings(embeddings, embedding_path)

    return embeddings


def compute_and_save_split_embeddings(
    split_dataframes: dict[str, pd.DataFrame],
    settings: dict[str, Any] | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Compute and cache embeddings for multiple split dataframes."""
    modeling_settings = settings or load_modeling_settings()
    split_embeddings: dict[str, np.ndarray] = {}

    for split_name, split_df in split_dataframes.items():
        split_embeddings[split_name] = get_or_compute_embeddings(
            split_df,
            split_name=split_name,
            settings=modeling_settings,
            overwrite=overwrite,
            show_progress=show_progress,
        )

    return split_embeddings


__all__ = [
    "compute_and_save_split_embeddings",
    "compute_embeddings_for_dataframe",
    "encode_texts_with_bert",
    "get_or_compute_embeddings",
    "get_text_column",
    "load_embedding_components",
    "load_embedding_settings",
    "load_embeddings",
    "mean_pool_hidden_states",
    "pool_embeddings",
    "resolve_device",
    "save_embeddings",
]