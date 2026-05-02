from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PRELABEL_DIR = PROJECT_ROOT / "data" / "data_prelabel"
OUTPUT_DIR = PROJECT_ROOT / "data" / "data_prelabel_predictions"

SUBTITLE_CODES_DIR = PROJECT_ROOT.parent / "subtitle_translation_en_to_hindi" / "codes"
if str(SUBTITLE_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(SUBTITLE_CODES_DIR))

from feature_extraction import FeatureExtractor
from prevalence_table import CATEGORIES, add_risk_flags


SOURCE_FILES = [
    "dialogue_metadata.csv",
    "dialogue_metadata_with_listener.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict subtitle translation risk for movie_social_re prelabel data.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_PRELABEL_DIR,
        help="Directory containing zipped prelabel movie folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where prediction CSVs will be written.",
    )
    return parser.parse_args()


def build_reason_columns(flagged: pd.DataFrame) -> pd.DataFrame:
    out = flagged.copy()
    risk_cols = [f"_risk_{cat}" for cat in CATEGORIES]

    def categories_for_row(row: pd.Series) -> str:
        active = [cat for cat in CATEGORIES if bool(row[f"_risk_{cat}"])]
        return "|".join(active)

    def confidence_for_row(row: pd.Series) -> str:
        parts: list[str] = []
        for cat in ["social", "register", "constraint", "terminology", "ambiguity"]:
            if bool(row.get(f"_risk_{cat}", False)):
                conf = str(row.get(f"_risk_{cat}_confidence", "low"))
                parts.append(f"{cat}:{conf}")
        return "|".join(parts)

    def cue_summary(row: pd.Series) -> str:
        cues: list[str] = []
        cue_map = {
            "idiomatic": ["cat1_idiom_match", "cat1_pv_opaque"],
            "pragmatic": [
                "cat2_discourse_markers",
                "cat2_negated_positive",
                "cat2_negated_negative",
                "cat2_caps_words",
            ],
            "social": [
                "_cue_social_request_form",
                "_cue_social_addressed_request",
                "_cue_social_addressed_command",
            ],
            "register": [
                "_cue_register_colloquial_strong",
                "_cue_register_formal_strong",
                "_cue_register_profanity_count",
                "_cue_register_elevated_count",
            ],
            "constraint": [
                "_cue_constraint_longish",
                "_cue_constraint_very_long",
                "_cue_constraint_dense",
                "_cue_constraint_terminology_heavy",
                "_cue_constraint_fragmented",
            ],
            "fragmentation": [
                "cat6_complete_sentence",
                "cat6_ellipsis_marker",
                "cat6_starts_lowercase",
                "cat6_ends_incomplete",
            ],
            "terminology": [
                "_cue_terminology_rare_strong",
                "_cue_terminology_entity_combo",
                "_cue_terminology_domain_combo",
                "_cue_terminology_multi_cap",
                "_cue_terminology_acronym",
            ],
            "ambiguity": [
                "_cue_ambiguity_deictic",
                "_cue_ambiguity_pronoun_heavy",
                "_cue_ambiguity_fragment",
                "_cue_ambiguity_modal",
                "_cue_ambiguity_underspecified_short",
            ],
        }
        for cat in CATEGORIES:
            if not bool(row[f"_risk_{cat}"]):
                continue
            for col in cue_map.get(cat, []):
                value = row.get(col)
                if pd.isna(value):
                    continue
                if isinstance(value, str) and value.strip():
                    cues.append(f"{col}={value}")
                elif bool(value):
                    cues.append(col)
        return "|".join(cues)

    out["pred_translation_error_possible"] = out["_any_risk"].astype(int)
    out["pred_risk_category_count"] = out[risk_cols].sum(axis=1).astype(int)
    out["pred_risk_categories"] = out.apply(categories_for_row, axis=1)
    out["pred_risk_confidence_summary"] = out.apply(confidence_for_row, axis=1)
    out["pred_risk_cues"] = out.apply(cue_summary, axis=1)
    return out


def load_csv_from_zip(zip_path: Path, inner_name: str) -> pd.DataFrame:
    movie_slug = zip_path.stem
    member = f"{movie_slug}/{inner_name}"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            df = pd.read_csv(handle)
    df = df.copy()
    df["movie"] = df.get("movie_name", movie_slug).fillna(movie_slug).astype(str)
    df["en"] = df["text"].fillna("").astype(str)
    df["line_number"] = range(1, len(df) + 1)
    df["source_zip"] = zip_path.name
    df["source_csv"] = inner_name
    df["source_row_index"] = range(len(df))
    return df


def predict_for_file(feature_extractor: FeatureExtractor, df: pd.DataFrame) -> pd.DataFrame:
    features = feature_extractor.extract_all_features(df[["movie", "en", "line_number"]].copy())
    flagged = add_risk_flags(features, version="revised")
    enriched = build_reason_columns(flagged)

    base_cols = [
        "movie",
        "line_number",
        "en",
        "pred_translation_error_possible",
        "pred_risk_category_count",
        "pred_risk_categories",
        "pred_risk_confidence_summary",
        "pred_risk_cues",
        "_any_risk",
        "_clean",
    ] + [f"_risk_{cat}" for cat in CATEGORIES]

    meta_cols = [
        "source_zip",
        "source_csv",
        "source_row_index",
        "movie_name",
        "movie_idx",
        "conversation_id",
        "utterance_id",
        "timestamp",
        "speaker_id",
        "speaker_name",
        "reply_to",
        "listener_id",
        "listener_name",
        "text",
    ]

    available_meta = [col for col in meta_cols if col in df.columns]
    available_base = [col for col in base_cols if col in enriched.columns]
    return pd.concat(
        [
            df[available_meta].reset_index(drop=True),
            enriched[available_base].reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(args.input_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found in {args.input_dir}")

    feature_extractor = FeatureExtractor()
    combined_outputs: list[pd.DataFrame] = []

    for zip_path in zip_paths:
        movie_slug = zip_path.stem
        movie_output_dir = args.output_dir / movie_slug
        movie_output_dir.mkdir(parents=True, exist_ok=True)

        for source_file in SOURCE_FILES:
            source_df = load_csv_from_zip(zip_path, source_file)
            pred_df = predict_for_file(feature_extractor, source_df)
            out_name = source_file.replace(".csv", "_risk_predictions.csv")
            pred_df.to_csv(movie_output_dir / out_name, index=False, encoding="utf-8-sig")
            combined_outputs.append(pred_df)
            risk_rate = pred_df["pred_translation_error_possible"].mean() * 100
            print(
                f"[saved] {movie_slug}/{out_name} | rows={len(pred_df)} "
                f"| predicted-risk={risk_rate:.1f}%"
            )

    combined = pd.concat(combined_outputs, ignore_index=True)
    combined.to_csv(
        args.output_dir / "all_prelabel_risk_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        combined.groupby(["source_zip", "source_csv"], as_index=False)
        .agg(
            total_rows=("pred_translation_error_possible", "size"),
            predicted_risk_rows=("pred_translation_error_possible", "sum"),
            predicted_risk_rate=("pred_translation_error_possible", "mean"),
        )
    )
    summary["predicted_risk_rate"] = summary["predicted_risk_rate"].round(4)
    summary.to_csv(
        args.output_dir / "all_prelabel_risk_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[saved] {args.output_dir / 'all_prelabel_risk_predictions.csv'}")
    print(f"[saved] {args.output_dir / 'all_prelabel_risk_summary.csv'}")


if __name__ == "__main__":
    main()
