"""
Movie-level risk-category prevalence table.

For each movie in sampled_features_df, computes the percentage of lines
flagged by each of 8 translation-risk categories, plus any_risk and clean.

Can be used two ways:

  1. As a module from a notebook:
       from prevalence_table import compute_prevalence, save_outputs, print_report
       table = compute_prevalence(sampled_features_df)
       save_outputs(table, '../results')
       print_report(table)

  2. As a standalone script (loads via stratified sampler):
       conda activate subtitle_translation
       python codes/prevalence_table.py
"""

import os, sys
import pandas as pd
import numpy as np
import re

sys.path.insert(0, os.path.dirname(__file__))

ROOT    = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR = os.path.join(ROOT, 'results')

CATEGORIES = [
    'idiomatic', 'pragmatic', 'social', 'register',
    'constraint', 'fragmentation', 'terminology', 'ambiguity',
]

DEICTIC_RE = re.compile(r"\b(this|that|these|those|here|there|now|then)\b", re.IGNORECASE)
PRONOUN_RE = re.compile(
    r"\b(i|me|we|us|he|him|she|her|it|they|them|this|that|these|those|someone|somebody|something|anyone|anybody|anything)\b",
    re.IGNORECASE,
)
PROFANITY_RE = re.compile(
    r"\b(fuck|fucking|shit|damn|goddamn|hell|asshole|bastard|bitch|crap|jerk|loser)\b",
    re.IGNORECASE,
)
FILLER_RE = re.compile(
    r"\b(uh|um|yeah|yep|nah|nope|yo|dude|bro|buddy|man|hey)\b",
    re.IGNORECASE,
)
ELEVATED_RE = re.compile(
    r"\b(with respect|your honor|ladies and gentlemen|permit me|therefore|hence|my lord|by all means|pursuant to|as requested)\b",
    re.IGNORECASE,
)
MULTI_CAP_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def _require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def _series_num(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(df[col], errors='coerce').fillna(fill)


def _series_text(df: pd.DataFrame) -> pd.Series:
    if 'en' not in df.columns:
        raise KeyError("add_risk_flags requires an 'en' column for revised text-aware category logic.")
    return df['en'].fillna('').astype(str)


# ── Risk flag definitions (exact per spec) ────────────────────────────────────

def _flag_idiomatic(df):
    return df['cat1_idiom_flag'] == 1

def _flag_pragmatic(df):
    return (
        (df['cat2_caps_words']       > 0) |
        (df['cat2_negated_positive'] > 0) |
        (df['cat2_negated_negative'] > 0) |
        (df['cat2_discourse_markers'] > 0)
    )

def _flag_social_legacy(df):
    return (
        (_series_num(df, 'cat3_vocative_count') > 0) |
        (_series_num(df, 'cat3_direct_command') == 1)
    )

def _flag_register_legacy(df):
    slang_flag     = _series_num(df, 'cat4_slang_count') >= 2
    formality      = pd.to_numeric(df['cat4_formality_score'], errors='coerce')
    formality_flag = formality.notna() & (formality > 5.2)
    return slang_flag | formality_flag

def _flag_constraint_legacy(df):
    return _series_num(df, 'cat5_length_risk') == 1

def _flag_fragmentation(df):
    incomplete = df['cat6_complete_sentence'] == 0
    marker     = (
        (df['cat6_starts_lowercase'] == 1) |
        (df['cat6_ellipsis_marker']  == 1)
    )
    return incomplete & marker

def _flag_terminology_legacy(df):
    return (
        (_series_num(df, 'cat7_compound_neologisms') > 0) |
        (_series_num(df, 'cat7_rare_word_count')     >= 2)
    )

def _flag_ambiguity_legacy(df):
    return (
        (_series_num(df, 'cat8_tone_ambiguous')  == 1) |
        (_series_num(df, 'cat8_implicit_subject') == 1)
    )


def _add_revised_logic_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        'cat2_exclamations', 'cat2_questions', 'cat2_ellipses',
        'cat3_vocative_count', 'cat3_you_count', 'cat3_hedge_count', 'cat3_direct_command',
        'cat4_slang_count', 'cat4_contraction_count', 'cat4_formality_score', 'cat4_avg_word_length',
        'cat5_src_words', 'cat5_src_chars', 'cat5_content_word_ratio', 'cat5_avg_syllables',
        'cat6_complete_sentence', 'cat6_ellipsis_marker', 'cat6_ends_incomplete',
        'cat7_ner_count', 'cat7_oov_rate', 'cat7_rare_word_count', 'cat7_unusual_cap_words', 'cat7_compound_neologisms',
        'cat8_brevity', 'cat8_is_single_word', 'cat8_modal_count', 'cat8_implicit_subject',
    ]
    _require_cols(df, needed, 'revised category logic')

    out = df.copy()
    text = _series_text(out)
    text_lower = text.str.lower()

    words = text.str.split().str.len().fillna(0)
    deictic = text.str.count(DEICTIC_RE)
    pronouns = text.str.count(PRONOUN_RE)
    profanity = text.str.count(PROFANITY_RE)
    fillers = text.str.count(FILLER_RE)
    elevated = text.str.count(ELEVATED_RE)
    multi_cap_entities = text.str.contains(MULTI_CAP_ENTITY_RE, regex=True)
    acronyms = text.str.contains(ACRONYM_RE, regex=True)

    brevity = _series_num(out, 'cat8_brevity') == 1
    single_word = _series_num(out, 'cat8_is_single_word') == 1
    modal = _series_num(out, 'cat8_modal_count') > 0
    implicit_subject = _series_num(out, 'cat8_implicit_subject') == 1
    fragment = (
        (_series_num(out, 'cat6_complete_sentence') == 0) |
        (_series_num(out, 'cat6_ellipsis_marker') == 1) |
        (_series_num(out, 'cat6_ends_incomplete') == 1) |
        (_series_num(out, 'cat2_ellipses') > 0)
    )
    pronoun_heavy = (pronouns >= 2) | ((pronouns >= 1) & brevity & (words <= 2))
    underspecified_short = brevity & (deictic > 0 | pronoun_heavy | modal | fragment)
    implicit_subject_contextual = implicit_subject & ((deictic > 0) | pronoun_heavy | modal | fragment)
    ambiguity_score = (
        implicit_subject_contextual.astype(int) * 3
        + (brevity & (deictic > 0)).astype(int) * 2
        + (brevity & pronoun_heavy).astype(int) * 2
        + (brevity & modal).astype(int) * 2
        + (brevity & fragment & ~single_word).astype(int) * 2
    )
    ambiguity_flag = implicit_subject_contextual | (
        brevity & (
            (deictic > 0) |
            pronoun_heavy |
            modal |
            (fragment & ~single_word)
        )
    )

    vocative = _series_num(out, 'cat3_vocative_count') > 0
    you = _series_num(out, 'cat3_you_count') > 0
    hedge = _series_num(out, 'cat3_hedge_count') > 0
    direct_command = _series_num(out, 'cat3_direct_command') == 1
    question = _series_num(out, 'cat2_questions') > 0
    exclaim = _series_num(out, 'cat2_exclamations') > 0
    social_request = hedge & (you | question)
    social_addressed_request = vocative & (hedge | question)
    social_addressed_command = direct_command & (vocative | you | exclaim)
    social_score = (
        social_request.astype(int) * 2
        + social_addressed_request.astype(int) * 2
        + social_addressed_command.astype(int) * 2
        + (vocative & you).astype(int)
    )
    social_flag = social_request | social_addressed_request | social_addressed_command

    slang = _series_num(out, 'cat4_slang_count')
    contractions = _series_num(out, 'cat4_contraction_count')
    formality = pd.to_numeric(out['cat4_formality_score'], errors='coerce')
    avg_word_len = _series_num(out, 'cat4_avg_word_length')
    src_words = _series_num(out, 'cat5_src_words')
    colloquial_strong = (
        (slang >= 2) |
        ((slang >= 1) & (contractions >= 1)) |
        (profanity > 0) |
        ((fillers > 0) & ((slang >= 1) | (contractions >= 1)))
    )
    formal_strong = (
        ((elevated > 0) & (src_words >= 5)) |
        (formality.notna() & (formality > 5.6) & (src_words >= 6) & (avg_word_len >= 4.4))
    )
    register_score = colloquial_strong.astype(int) * 2 + formal_strong.astype(int) * 2
    register_flag = colloquial_strong | formal_strong

    src_chars = _series_num(out, 'cat5_src_chars')
    content_ratio = _series_num(out, 'cat5_content_word_ratio')
    avg_syllables = _series_num(out, 'cat5_avg_syllables')
    rare_words = _series_num(out, 'cat7_rare_word_count')
    ner_count = _series_num(out, 'cat7_ner_count')
    compound_terms = _series_num(out, 'cat7_compound_neologisms')
    longish = (src_words >= 14) | (src_chars >= 95)
    very_long = (src_words >= 18) | (src_chars >= 115)
    dense = (content_ratio >= 0.68) | (avg_syllables >= 1.7)
    terminology_heavy = (rare_words >= 2) | (compound_terms > 0) | (ner_count >= 2)
    fragmented = (
        (_series_num(out, 'cat6_complete_sentence') == 0) |
        (_series_num(out, 'cat6_ellipsis_marker') == 1) |
        (_series_num(out, 'cat6_ends_incomplete') == 1)
    )
    constraint_score = (
        very_long.astype(int) * 3
        + (longish & dense).astype(int) * 2
        + (longish & terminology_heavy).astype(int) * 2
        + (longish & fragmented).astype(int) * 2
    )
    constraint_flag = very_long | (longish & (dense | terminology_heavy | fragmented))

    oov_rate = pd.to_numeric(out['cat7_oov_rate'], errors='coerce').fillna(0)
    unusual_caps = _series_num(out, 'cat7_unusual_cap_words')
    rare_strong = rare_words >= 3
    entity_combo = (
        ((ner_count >= 2) & (rare_words >= 1) & ((unusual_caps > 0) | multi_cap_entities | acronyms)) |
        ((ner_count >= 3) & (rare_words >= 1))
    )
    domain_combo = (
        (compound_terms > 0) |
        ((rare_words >= 2) & ((unusual_caps > 0) | (ner_count >= 1) | acronyms)) |
        ((oov_rate >= 0.25) & (rare_words >= 2))
    )
    terminology_score = rare_strong.astype(int) * 2 + entity_combo.astype(int) * 2 + domain_combo.astype(int) * 2
    terminology_flag = rare_strong | entity_combo | domain_combo

    out['_cue_ambiguity_deictic'] = (deictic > 0)
    out['_cue_ambiguity_pronoun_heavy'] = pronoun_heavy
    out['_cue_ambiguity_fragment'] = fragment
    out['_cue_ambiguity_modal'] = modal
    out['_cue_ambiguity_underspecified_short'] = underspecified_short
    out['_risk_ambiguity_score'] = ambiguity_score
    out['_risk_ambiguity_confidence'] = np.select(
        [ambiguity_score >= 4, ambiguity_score >= 2],
        ['high', 'medium'],
        default='low',
    )

    out['_cue_social_request_form'] = social_request
    out['_cue_social_addressed_request'] = social_addressed_request
    out['_cue_social_addressed_command'] = social_addressed_command
    out['_risk_social_score'] = social_score
    out['_risk_social_confidence'] = np.select(
        [social_score >= 4, social_score >= 2],
        ['high', 'medium'],
        default='low',
    )

    out['_cue_register_colloquial_strong'] = colloquial_strong
    out['_cue_register_formal_strong'] = formal_strong
    out['_cue_register_profanity_count'] = profanity
    out['_cue_register_filler_count'] = fillers
    out['_cue_register_elevated_count'] = elevated
    out['_risk_register_score'] = register_score
    out['_risk_register_confidence'] = np.select(
        [register_score >= 4, register_score >= 2],
        ['high', 'medium'],
        default='low',
    )

    out['_cue_constraint_longish'] = longish
    out['_cue_constraint_very_long'] = very_long
    out['_cue_constraint_dense'] = dense
    out['_cue_constraint_terminology_heavy'] = terminology_heavy
    out['_cue_constraint_fragmented'] = fragmented
    out['_risk_constraint_score'] = constraint_score
    out['_risk_constraint_confidence'] = np.select(
        [constraint_score >= 4, constraint_score >= 2],
        ['high', 'medium'],
        default='low',
    )

    out['_cue_terminology_rare_strong'] = rare_strong
    out['_cue_terminology_entity_combo'] = entity_combo
    out['_cue_terminology_domain_combo'] = domain_combo
    out['_cue_terminology_multi_cap'] = multi_cap_entities
    out['_cue_terminology_acronym'] = acronyms
    out['_risk_terminology_score'] = terminology_score
    out['_risk_terminology_confidence'] = np.select(
        [terminology_score >= 4, terminology_score >= 2],
        ['high', 'medium'],
        default='low',
    )

    out['_risk_social'] = social_flag.fillna(False)
    out['_risk_register'] = register_flag.fillna(False)
    out['_risk_constraint'] = constraint_flag.fillna(False)
    out['_risk_terminology'] = terminology_flag.fillna(False)
    out['_risk_ambiguity'] = ambiguity_flag.fillna(False)
    return out

_LEGACY_FLAG_FNS = {
    'idiomatic':    _flag_idiomatic,
    'pragmatic':    _flag_pragmatic,
    'social':       _flag_social_legacy,
    'register':     _flag_register_legacy,
    'constraint':   _flag_constraint_legacy,
    'fragmentation': _flag_fragmentation,
    'terminology':  _flag_terminology_legacy,
    'ambiguity':    _flag_ambiguity_legacy,
}


def add_risk_flags(df: pd.DataFrame, version: str = 'revised') -> pd.DataFrame:
    """Add one boolean column per category plus any_risk / clean. Returns a copy."""
    if version not in {'legacy', 'revised'}:
        raise ValueError("version must be 'legacy' or 'revised'")

    out = df.copy()
    if version == 'legacy':
        for cat, fn in _LEGACY_FLAG_FNS.items():
            out[f'_risk_{cat}'] = fn(out).fillna(False)
    else:
        for cat, fn in _LEGACY_FLAG_FNS.items():
            if cat in {'social', 'register', 'constraint', 'terminology', 'ambiguity'}:
                continue
            out[f'_risk_{cat}'] = fn(out).fillna(False)
        out = _add_revised_logic_columns(out)
    risk_cols      = [f'_risk_{c}' for c in CATEGORIES]
    out['_any_risk'] = out[risk_cols].any(axis=1)
    out['_clean']    = ~out['_any_risk']
    return out


# ── Prevalence computation ────────────────────────────────────────────────────

def compute_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movie-level risk-category prevalence table.

    Args:
        df: DataFrame with a 'movie' column and all cat1_–cat8_ feature columns.

    Returns:
        DataFrame indexed by movie, columns:
            idiomatic, pragmatic, social, register, constraint, fragmentation,
            terminology, ambiguity, any_risk, clean  — all in % (1 decimal)
            total_lines — raw count
        Plus a bottom row "Overall" (weighted corpus-wide average).
    """
    flagged = add_risk_flags(df)

    # ── Per-movie stats ───────────────────────────────────────────────────────
    rows = {}
    small_movies = []

    for movie, grp in flagged.groupby('movie'):
        n = len(grp)
        if n < 50:
            small_movies.append((movie, n))

        row = {}
        for cat in CATEGORIES:
            row[cat] = grp[f'_risk_{cat}'].mean() * 100
        row['any_risk']    = grp['_any_risk'].mean() * 100
        row['clean']       = grp['_clean'].mean()    * 100
        row['total_lines'] = n
        rows[movie] = row

    table = pd.DataFrame(rows).T
    table.index.name = 'movie'

    # ── Overall row (weighted by total_lines) ─────────────────────────────────
    weights   = table['total_lines']
    total_n   = weights.sum()
    pct_cols  = CATEGORIES + ['any_risk', 'clean']
    overall   = {col: (table[col] * weights).sum() / total_n for col in pct_cols}
    overall['total_lines'] = total_n
    table.loc['Overall'] = overall

    # Round display columns to 1 decimal; keep total_lines as int
    for col in pct_cols:
        table[col] = table[col].round(1)
    table['total_lines'] = table['total_lines'].astype(int)

    # ── Column order ──────────────────────────────────────────────────────────
    table = table[CATEGORIES + ['any_risk', 'clean', 'total_lines']]

    if small_movies:
        print(f"\n⚠  Movies with fewer than 50 lines (results unreliable):")
        for m, n in small_movies:
            print(f"   {m}: {n} lines")

    return table


# ── Output helpers ────────────────────────────────────────────────────────────

def _to_markdown(table: pd.DataFrame) -> str:
    pct_cols = CATEGORIES + ['any_risk', 'clean']
    header   = ['movie'] + pct_cols + ['total_lines']
    sep      = ['---'] + ['---:'] * (len(pct_cols) + 1)

    lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(sep)    + ' |',
    ]
    for idx, row in table.iterrows():
        vals = [str(idx)]
        for col in pct_cols:
            vals.append(f"{row[col]:.1f}")
        vals.append(str(int(row['total_lines'])))
        lines.append('| ' + ' | '.join(vals) + ' |')

    return '\n'.join(lines)


def save_outputs(table: pd.DataFrame, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'prevalence_table.csv')
    table.to_csv(csv_path)
    print(f"Saved → {csv_path}")

    md_path  = os.path.join(out_dir, 'prevalence_table.md')
    md_text  = (
        "# Movie-Level Risk-Category Prevalence\n\n"
        "_Values are % of lines in each movie flagged by each risk category._\n\n"
        + _to_markdown(table) + "\n"
    )
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"Saved → {md_path}")


def print_report(table: pd.DataFrame):
    movie_rows = table.drop(index='Overall', errors='ignore')
    pct_cols   = CATEGORIES + ['any_risk', 'clean']

    print("\n" + "=" * 90)
    print("PREVALENCE TABLE (% of lines per movie flagged by each risk category)")
    print("=" * 90)
    print(_to_markdown(table))

    print("\n" + "=" * 60)
    print("CATEGORY HIGHLIGHTS")
    print("=" * 60)

    category_totals = {}
    for cat in CATEGORIES:
        top_movie = movie_rows[cat].idxmax()
        top_pct   = movie_rows[cat].max()
        overall   = table.loc['Overall', cat] if 'Overall' in table.index else float('nan')
        category_totals[cat] = overall
        print(f"  {cat:<16}: highest in '{top_movie}' ({top_pct:.1f}%)  [corpus avg {overall:.1f}%]")

    top_cat = max(category_totals, key=category_totals.get)
    print(f"\n  Most common risk category overall: '{top_cat}' ({category_totals[top_cat]:.1f}%)")
    print("=" * 60)


# ── Standalone entry point ────────────────────────────────────────────────────

def _load_sampled_features():
    """Load via stratified sampler — use when running as a script."""
    from stratified_sampler import sample_from_precomputed, SELECTED_MOVIES
    print(f"Loading opus_feature_matrix_v2.csv ...")
    feat_path = os.path.join(ROOT, 'results', 'opus_feature_matrix_v2.csv')
    fm = pd.read_csv(feat_path)
    fm['line_idx'] = fm.index
    print(f"  {len(fm):,} rows loaded")

    # Use ALL lines from selected movies (not just the 200-line stratified sample)
    # so prevalence percentages reflect the true corpus distribution.
    movie_titles = [m[0] for m in SELECTED_MOVIES]
    subset = fm[fm['movie'].isin(movie_titles)].copy()
    print(f"  {len(subset):,} lines across {subset['movie'].nunique()} movies: "
          f"{sorted(subset['movie'].unique())}")
    return subset


if __name__ == '__main__':
    df = _load_sampled_features()
    table = compute_prevalence(df)
    print_report(table)
    save_outputs(table)
