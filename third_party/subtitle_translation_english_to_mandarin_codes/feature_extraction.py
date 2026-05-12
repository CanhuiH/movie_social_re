"""
Feature Extraction v2 — Corrected & Optimized
================================================
Fixes applied from code review:

 1. Compositionality score: normalize by sentence length, add length-corrected variant
 2. Idiom frequency: use actual wordfreq corpus frequency, not list position
 3. Phrasal verbs: split into opaque (non-compositional) vs transparent
 4. POSITIVE_ADJECTIVES: fixed — separated positive vs negative polar adjectives
 5. Discourse markers: require sentence-initial or standalone position to reduce FP
 6. Vocative titles: use spaCy dependency parse for address detection, not just token match
 7. Formality score: guard against short lines (< 5 words get NaN)
 8. Voice shift: reset window at movie boundaries (handled in extract_all_features)
 9. Length risk: use industry-standard 42 chars/line threshold
10. OPUS tokenization: clean spacing artifacts before parsing
11. Implicit subject: exclude imperative sentences from this feature
12. Error handling: try/except per-movie with logging
13. Redundant spaCy calls: single parse per line via _parse_once()
14. Back-translation warnings: fixed max_length / max_new_tokens conflict

Note:
  This module intentionally preserves the raw numeric feature inventory.
  The human-audit-driven revisions to category-level risk triggering now live
  downstream in `codes/prevalence_table.py`, where the raw features are combined
  into stricter, more interpretable category flags.

Usage:
    from feature_extraction_v2 import FeatureExtractor
    fe = FeatureExtractor()  # loads models once
    features_df = fe.extract_all_features(df)  # df must have 'en' column
"""

import re
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

for resource in ['wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger']:
    nltk.download(resource, quiet=True)

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordfreq import word_frequency, zipf_frequency

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# LEXICONS
# =============================================================================

IDIOMS = [
    # Body idioms
    "kick the bucket", "bite the bullet", "break a leg", "hit the nail on the head",
    "cost an arm and a leg", "put your foot in your mouth", "hit below the belt",
    "keep your fingers crossed", "face the music", "bite the hand that feeds you",
    "bite off more than you can chew", "all ears", "lose your touch",
    # Animal idioms
    "let the cat out of the bag", "let sleeping dogs lie", "barking up the wrong tree",
    "kill two birds with one stone", "once bitten twice shy", "snake in the grass",
    "straight from the horse's mouth", "hold your horses", "like two peas in a pod",
    "needle in a haystack", "open a can of worms",
    # Directional / positional
    "under the weather", "under someone's thumb", "under fire", "throw under the bus",
    "up in the air", "on thin ice", "on the fence", "on the same page",
    "on the ball", "out of the blue", "out of the loop",
    # Common idiomatic expressions
    "piece of cake", "spill the beans", "once in a blue moon", "at the drop of a hat",
    "pull someone's leg", "steal someone's thunder", "burn bridges", "cut corners",
    "hit the sack", "call it a day", "get out of hand", "time flies",
    "back to the drawing board", "beat around the bush",
    "get a taste of your own medicine", "wrap your head around",
    "jump on the bandwagon", "knock it out of the park", "throw in the towel",
    "through thick and thin", "your guess is as good as mine",
    "add fuel to the fire", "blow off steam", "cut to the chase",
    "every cloud has a silver lining", "get the hang of it",
    "give someone the cold shoulder", "give the benefit of the doubt",
    "hit the ground running", "make a long story short", "miss the boat",
    "put all your eggs in one basket", "see the writing on the wall",
    "stab in the back", "stand your ground", "the last straw",
    "twist someone's arm", "a blessing in disguise", "add insult to injury",
    "at the end of the day", "blow the whistle", "break the ice",
    "burning the midnight oil", "close but no cigar", "dead ringer",
    "devil's advocate", "don't cry over spilled milk", "drop the ball",
    "fill someone's shoes", "fit as a fiddle", "get cold feet",
    "give up the ghost", "go down in flames", "go the extra mile",
    "hang in there", "in hot water", "jump to conclusions",
    "keep someone in the loop", "know the ropes", "leave no stone unturned",
    "let someone off the hook", "light at the end of the tunnel",
    "make ends meet", "miss the mark", "more than meets the eye",
    "no pain no gain", "pass the buck", "pull the plug",
    "put the cart before the horse", "read between the lines",
    "run out of steam", "set the record straight", "show your true colors",
    "take the bull by the horns", "the bottom line",
    "throw caution to the wind", "tie up loose ends", "turn over a new leaf",
    "wear your heart on your sleeve",
    "you can't have your cake and eat it too",
    "milk it", "bite the dust", "cold turkey", "cream of the crop",
    "curiosity killed the cat", "go cold turkey",
    "it takes two to tango", "keep a stiff upper lip",
    "take a rain check", "the ball is in your court",
    "the best of both worlds", "the tip of the iceberg",
    "think outside the box", "under the table", "the whole nine yards",
]

# FIX #3: Split phrasal verbs into OPAQUE (non-compositional, high MT risk)
# and TRANSPARENT (literal meaning obvious, low MT risk).
# Only opaque ones are reliable translation-difficulty signals.
PHRASAL_VERBS_OPAQUE = [
    "milk it", "call off", "pull off", "give in", "give up",
    "figure out", "turn out", "take on", "take over", "take up",
    "put up with", "put out", "put off", "put down", "hold up",
    "hold off", "hold back", "look into", "look after", "look down on",
    "look up to", "bring up", "bring about", "bring down", "carry out",
    "come across", "come around", "come off", "come through", "come up with",
    "cut back", "cut down", "cut out", "deal with", "do away with",
    "drop out", "fall for", "fall through", "find out",
    "get along", "get around", "get away", "get by",
    "get into", "get over", "get rid of", "get through",
    "give away", "go for", "go off", "go over", "go through",
    "hand out", "hang out", "keep up", "lay off", "let down", "let off",
    "make out", "make up", "move on", "pass away", "pass out",
    "pay off", "point out", "run into", "run out", "run over",
    "set off", "set out", "settle down", "show off", "show up",
    "shut down", "sort out", "stand out", "stick out",
    "think over", "throw away", "try out",
    "turn down", "turn up", "use up", "watch out", "wear out", "work out",
    "write off", "blow up", "break out", "break up",
    "burn out", "catch up", "clean up", "clear up",
    "close down", "crack down", "end up", "fill in",
    "follow up", "kick in", "kick out", "knock out",
    "lay out", "leave out", "line up", "live up to",
    "lock out", "mix up", "narrow down", "pan out", "phase out",
    "pull back", "pull out", "pull through", "put forward", "ramp up",
    "reach out", "roll out", "rule out", "sell out", "sign up",
    "speak out", "speak up", "step back", "step down", "step in",
    "step up", "sum up", "take back", "take care of", "take down", "take in",
    "track down", "wind down", "wind up", "wrap up",
    "blow off", "bottle up", "brush off", "butt in", "cave in",
    "chicken out", "chip in", "clam up", "cop out", "count on",
    "drag on", "draw out", "ease up", "egg on",
    "face up to", "fade out", "fit in", "flare up",
    "freak out", "frown upon", "gang up", "gear up", "hit on",
    "iron out", "lash out", "lean on", "level off",
    "liven up", "long for", "mess up", "nod off", "open up", "own up",
    "pile up", "pipe down", "pitch in", "play down", "play up",
    "prop up", "push through", "read up", "ring up",
    "rise up", "round up", "scrape by", "snap out of",
    "square up", "stand by", "start over", "stick up for",
    "stir up", "straighten out", "string along",
    "stumble upon", "suit up", "take apart",
    "tighten up", "tone down", "toss out", "touch on",
    "trip up", "tune out", "vote out",
    "walk out", "warm up", "weed out", "well up",
]

# Transparent phrasal verbs — tracked separately, lower weight
PHRASAL_VERBS_TRANSPARENT = [
    "pick up", "put on", "set up", "take off", "get up",
    "sit down", "wake up", "go away", "come back", "go ahead",
    "come out", "get out", "hold on", "hang up", "shut up",
    "slow down", "speed up", "cool down", "heat up", "light up",
    "log in", "lie down", "stand up", "switch off", "switch on",
    "turn off", "turn on", "turn around", "drop by", "drop off",
    "fall apart", "fall back", "fall behind", "get down",
    "get off", "get on", "get together", "give back", "give out",
    "grow up", "hand in", "hang on", "keep on",
    "let out", "pay back", "run away", "save up",
    "calm down", "quiet down", "pop in", "dive in", "jump in",
    "lay down", "stretch out", "sweep up", "type up",
]

# FIX #4: Separate positive and negative polar adjectives
POSITIVE_ADJECTIVES = {
    'good', 'great', 'fine', 'nice', 'easy', 'smart', 'bright', 'clear',
    'simple', 'obvious', 'sure', 'okay', 'perfect', 'right', 'true', 'fair',
    'wonderful', 'excellent', 'amazing', 'fantastic', 'brilliant',
}
NEGATIVE_ADJECTIVES = {
    'bad', 'wrong', 'stupid', 'terrible', 'horrible', 'awful', 'dumb',
    'cheap', 'ugly', 'boring', 'pathetic', 'lousy', 'miserable',
}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nor', 'nothing',
                  'nobody', 'nowhere', 'hardly', 'barely', 'scarcely'}

# FIX #5: Discourse markers — only sentence-initial ones are reliable signals
# Multi-word markers kept as-is (already positional by nature)
DISCOURSE_MARKERS_INITIAL = {
    'well', 'look', 'listen', 'hey', 'wait', 'okay', 'so', 'now',
    'actually', 'basically', 'honestly', 'frankly',
    'apparently', 'supposedly', 'allegedly',
}
DISCOURSE_MARKERS_MULTIWORD = [
    'i mean', 'you know', 'that is', 'in other words', 'by that i mean',
]

# FIX #6: Titles — separate vocative-likely from ambiguous words
FORMAL_TITLES = {'sir', "ma'am", 'madam', 'mister', 'mr', 'mrs', 'ms', 'miss',
                 'dr', 'doctor', 'professor', 'officer', 'detective', 'captain',
                 'sergeant', 'general', 'colonel', 'lieutenant', 'agent',
                 'your honor', 'your majesty', 'chairman', 'president',
                 'reverend', 'bishop'}

# "man", "boy", "girl", "son" removed — too ambiguous without dep parse
INFORMAL_TITLES = {'dude', 'buddy', 'pal', 'bro', 'babe', 'baby',
                   'honey', 'sweetheart', 'darling', 'love', 'dear', 'kid',
                   'guys', 'ladies', 'folks', 'mate',
                   'champ', 'chief', 'boss', 'sport'}

ALL_TITLES = FORMAL_TITLES | INFORMAL_TITLES

HEDGES = ['could you', 'would you', 'would you mind', 'do you mind',
          'please', 'if you could', 'if you would', "if you don't mind",
          'i was wondering', 'i wonder if', "i'm sorry",
          'excuse me', 'pardon me', 'with respect']

SLANG = {
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme', 'gimme',
    "ain't", "y'all", 'yeah', 'yep', 'nope', 'nah', 'uh-huh', 'yup',
    'cool', 'awesome', 'dude', 'bro', 'lol', 'omg',
    'wtf', 'hell', 'damn', 'crap', 'chill', 'freak', 'jerk', 'loser',
    'screw', 'suck', 'sucks', 'blows', 'bucks', 'dough', 'grand',
    'cop', 'buzz', 'booze', 'wasted', 'hammered', 'trashed', 'stoked',
    'psyched', 'pumped', 'bummed', 'bummer', 'flaky', 'sketchy', 'shady',
    'legit', 'lit', 'sick', 'dope', 'fire', 'savage', 'wack', 'lame',
    'wimp', 'wuss', 'punk', 'creep', 'sleaze', 'slob',
    'freaking', 'effing',
}

MODAL_VERBS = {'could', 'might', 'should', 'would', 'may', 'can', 'must',
               'ought', 'shall', 'need', 'dare'}

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

CONTRACTION_PATTERN = re.compile(
    r"(\b\w+'[a-z]{1,2}\b|\b[a-z]+'[a-z]{1,2}\b)",
    re.IGNORECASE
)

OOV_THRESHOLD = 1e-6
RARE_THRESHOLD = 1e-5
VADER_NEUTRAL_THRESHOLD = 0.15


# =============================================================================
# FIX #10: OPUS tokenization cleanup
# =============================================================================

def clean_opus_tokenization(text: str) -> str:
    """Fix OPUS spacing artifacts: 'I know Tyler Durden .' -> 'I know Tyler Durden.'"""
    text = re.sub(r'\s+([.!?,;:\'\")\]}])', r'\1', text)
    text = re.sub(r'([(\[{])\s+', r'\1', text)
    return text.strip()


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class FeatureExtractor:
    """
    Extracts 47 source-side features across 8 error categories.

    Usage:
        fe = FeatureExtractor()
        df_features = fe.extract_all_features(df)  # df must have 'en' column
    """

    def __init__(self, load_backtranslation=False):
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['textcat'])
        except OSError:
            import subprocess
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm', disable=['textcat'])

        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer
        self.minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.vader = SentimentIntensityAnalyzer()

        # Build lemmatized lookup tables
        logger.info("Building idiom/PV lookup tables...")
        self.IDIOM_LOOKUP = {self._lemmatize_phrase(i): i for i in IDIOMS}
        self.PV_OPAQUE_LOOKUP = {self._lemmatize_phrase(pv): pv for pv in PHRASAL_VERBS_OPAQUE}
        self.PV_TRANSPARENT_LOOKUP = {self._lemmatize_phrase(pv): pv for pv in PHRASAL_VERBS_TRANSPARENT}

        # FIX #2: Use actual corpus frequency instead of list position
        self.IDIOM_FREQ = {}
        for idiom in IDIOMS:
            key = self._lemmatize_phrase(idiom)
            # Average zipf frequency of content words in the idiom
            words = [w for w in idiom.split() if len(w) > 2]
            if words:
                avg_zipf = np.mean([zipf_frequency(w, 'en') for w in words])
                # Invert: common words in idiom = more likely to be misread as literal
                self.IDIOM_FREQ[key] = avg_zipf
            else:
                self.IDIOM_FREQ[key] = 3.0  # default mid-range

        self.bt_model = None
        self.bt_tokenizer = None
        if load_backtranslation:
            self._load_backtranslation()

        logger.info("FeatureExtractor ready.")

    def _load_backtranslation(self):
        """Load ZH->EN back-translation model."""
        from transformers import MarianMTModel, MarianTokenizer
        model_name = 'Helsinki-NLP/opus-mt-zh-en'
        logger.info(f"Loading back-translation model: {model_name}")
        self.bt_tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.bt_model = MarianMTModel.from_pretrained(model_name)

    def _lemmatize_phrase(self, phrase: str) -> tuple:
        doc = self.nlp(phrase.lower())
        return tuple(t.lemma_ for t in doc if t.is_alpha)

    def _normalize_tokens(self, text: str) -> list:
        doc = self.nlp(text.lower())
        return [t.lemma_ for t in doc if t.is_alpha]

    @staticmethod
    def _get_ngrams(tokens: list, n: int) -> list:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    # ─── Category 1: Idiomatic Expression Failure ──────────────────────────

    def cat1_lexicon_features(self, text: str) -> dict:
        """
        Idiom and phrasal verb lexicon matching.

        Returns:
          cat1_idiom_match:     matched idiom string or None
          cat1_idiom_freq:      corpus frequency of idiom words (higher = more common words,
                                meaning higher risk of literal misinterpretation)
          cat1_pv_opaque:       matched opaque phrasal verb or None
          cat1_pv_transparent:  matched transparent phrasal verb or None
          cat1_idiom_flag:      True if any idiom or opaque PV found
        """
        tokens = self._normalize_tokens(text)
        idiom_match, idiom_freq = None, float('nan')
        pv_opaque, pv_transparent = None, None

        for n in range(2, 7):
            if n > len(tokens):
                break
            for gram in self._get_ngrams(tokens, n):
                if idiom_match is None and gram in self.IDIOM_LOOKUP:
                    idiom_match = self.IDIOM_LOOKUP[gram]
                    idiom_freq = self.IDIOM_FREQ.get(gram, float('nan'))
                if pv_opaque is None and gram in self.PV_OPAQUE_LOOKUP:
                    pv_opaque = self.PV_OPAQUE_LOOKUP[gram]
                if pv_transparent is None and gram in self.PV_TRANSPARENT_LOOKUP:
                    pv_transparent = self.PV_TRANSPARENT_LOOKUP[gram]
            if idiom_match and pv_opaque:
                break

        return {
            'cat1_idiom_match':     idiom_match,
            'cat1_idiom_freq':      idiom_freq,
            'cat1_pv_opaque':       pv_opaque,
            'cat1_pv_transparent':  pv_transparent,
            # FIX #3: Only opaque PVs count toward idiom_flag
            'cat1_idiom_flag':      idiom_match is not None or pv_opaque is not None,
        }

    def cat1_compositionality(self, texts: list, batch_size: int = 128) -> list:
        """
        FIX #1: Compositionality with length normalization.

        Returns a dict with two scores per line:
          - cat1_compositionality_raw:  cosine(sent_emb, mean_word_emb) — original
          - cat1_compositionality:      length-normalized via residuals from a linear fit

        Low score → figurative / non-literal. NaN for lines < 3 words.
        """
        eligible_idx, eligible_texts, eligible_words, eligible_lengths = [], [], [], []
        for i, text in enumerate(texts):
            words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
            if len(words) >= 3:
                eligible_idx.append(i)
                eligible_texts.append(text)
                eligible_words.append(words)
                eligible_lengths.append(len(words))

        if not eligible_texts:
            return [float('nan')] * len(texts)

        sent_embs = self.minilm.encode(eligible_texts, batch_size=batch_size,
                                        show_progress_bar=True)

        unique_words = list({w for words in eligible_words for w in words})
        logger.info(f'Encoding {len(unique_words):,} unique words...')
        word_emb_batch = self.minilm.encode(unique_words, batch_size=batch_size,
                                             show_progress_bar=True)
        word_emb_map = dict(zip(unique_words, word_emb_batch))

        raw_scores = []
        for sent_emb, words in zip(sent_embs, eligible_words):
            word_mean = np.stack([word_emb_map[w] for w in words]).mean(axis=0)
            score = float(cosine_similarity([sent_emb], [word_mean])[0][0])
            raw_scores.append(score)

        # FIX #1: Length-normalize via linear regression residuals
        # Longer sentences naturally have lower compositionality — remove that trend
        raw_arr = np.array(raw_scores)
        len_arr = np.array(eligible_lengths, dtype=float)
        # Fit: raw_score ~ a * log(length) + b
        log_len = np.log(len_arr)
        coeffs = np.polyfit(log_len, raw_arr, 1)
        predicted = np.polyval(coeffs, log_len)
        normalized = raw_arr - predicted  # residual: negative = more figurative than expected

        result_raw = [float('nan')] * len(texts)
        result_norm = [float('nan')] * len(texts)
        for idx, raw, norm in zip(eligible_idx, raw_scores, normalized):
            result_raw[idx] = raw
            result_norm[idx] = float(norm)

        return result_raw, result_norm

    # ─── Category 2: Pragmatic Meaning Loss ────────────────────────────────

    def cat2_features(self, text: str) -> dict:
        """Category 2: Pragmatic Meaning Loss features."""
        vs = self.vader.polarity_scores(text)

        exclamations = text.count('!')
        questions = text.count('?')
        ellipses = len(re.findall(r'\.{2,}', text))

        # ALL-CAPS words (length >= 2 to avoid 'I', 'A')
        caps_words = sum(1 for w in text.split()
                         if w.isupper() and len(w) >= 2 and w.isalpha())

        # FIX #5: Discourse markers — sentence-initial only for single-word markers
        text_lower = text.lower().strip()
        first_word = text_lower.split()[0] if text_lower.split() else ''
        dm_initial = int(first_word.rstrip('.,!?;:') in DISCOURSE_MARKERS_INITIAL)
        dm_multiword = sum(1 for m in DISCOURSE_MARKERS_MULTIWORD
                           if m in text_lower)
        dm_count = dm_initial + dm_multiword

        # Negation
        tokens_lower = word_tokenize(text_lower)
        neg_count = sum(1 for t in tokens_lower if t in NEGATION_WORDS)

        # FIX #4: Separate negated-positive and negated-negative (litotes)
        negated_pos, negated_neg = 0, 0
        for i, tok in enumerate(tokens_lower[:-1]):
            if tok in NEGATION_WORDS:
                next_tok = tokens_lower[i + 1]
                if next_tok in POSITIVE_ADJECTIVES:
                    negated_pos += 1
                elif next_tok in NEGATIVE_ADJECTIVES:
                    negated_neg += 1

        return {
            'cat2_vader_compound':   vs['compound'],
            'cat2_vader_neg':        vs['neg'],
            'cat2_vader_pos':        vs['pos'],
            'cat2_exclamations':     exclamations,
            'cat2_questions':        questions,
            'cat2_ellipses':         ellipses,
            'cat2_caps_words':       caps_words,
            'cat2_discourse_markers': dm_count,
            'cat2_dm_rate':          dm_count / max(len(text_lower.split()), 1),  # FIX #15: normalized by word count
            'cat2_negations':        neg_count,
            'cat2_negated_positive': negated_pos,
            'cat2_negated_negative': negated_neg,  # NEW: litotes detection
        }

    # ─── Category 3: Social Hierarchy & Politeness Mismatch ────────────────

    def cat3_features(self, doc) -> dict:
        """
        Category 3: Social Hierarchy & Politeness features.
        FIX #6: Use spaCy dependency parse for vocative detection.
        Accepts a spaCy Doc (from single-parse pipeline).
        """
        text_lower = doc.text.lower()
        tokens_lower = [t.text.lower() for t in doc]

        # FIX #6: Check for vocative/appositive dependency, not just token presence
        # spaCy marks vocatives as 'npadvmod' or 'appos' in some cases,
        # but more reliably: check if title word is not a subject/object
        formal_count, informal_count = 0, 0
        for tok in doc:
            tok_lower = tok.text.lower()
            if tok_lower in FORMAL_TITLES:
                # Only count if it's likely a vocative (not a regular noun role)
                if tok.dep_ not in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'):
                    formal_count += 1
                elif tok.dep_ in ('ROOT', 'appos', 'npadvmod'):
                    formal_count += 1
            if tok_lower in INFORMAL_TITLES:
                if tok.dep_ not in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'):
                    informal_count += 1
                elif tok.dep_ in ('ROOT', 'appos', 'npadvmod'):
                    informal_count += 1

        vocative_count = formal_count + informal_count
        if formal_count > 0:
            title_type = 'formal'
        elif informal_count > 0:
            title_type = 'informal'
        else:
            title_type = 'none'

        # 2nd person pronouns
        you_count = sum(1 for t in doc
                        if t.text.lower() in {'you', 'your', 'yours', 'yourself', 'yourselves'})

        # Imperative: ROOT is VERB with no nsubj
        is_imperative = False
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                has_subject = any(child.dep_ in ('nsubj', 'nsubjpass')
                                  for child in token.children)
                if not has_subject:
                    is_imperative = True
                break

        # Hedging
        hedge_count = sum(1 for h in HEDGES if h in text_lower)

        is_direct_command = is_imperative and hedge_count == 0

        return {
            'cat3_vocative_count':  vocative_count,
            'cat3_title_type':      title_type,
            'cat3_you_count':       you_count,
            'cat3_imperative':      int(is_imperative),
            'cat3_hedge_count':     hedge_count,
            'cat3_direct_command':  int(is_direct_command),
        }

    # ─── Category 4: Register & Character Voice Mismatch ───────────────────

    def cat4_features(self, text: str) -> dict:
        """
        Category 4: Register & Character Voice features.
        FIX #7: Formality score returns NaN for lines < 5 words.
        """
        words = [w for w in re.findall(r'[a-zA-Z]+', text.lower())]

        if not words:
            return {
                'cat4_type_token_ratio':  float('nan'),
                'cat4_avg_word_length':   float('nan'),
                'cat4_formality_score':   float('nan'),
                'cat4_slang_count':       0,
                'cat4_contraction_count': 0,
            }

        ttr = len(set(words)) / len(words)
        avg_len = np.mean([len(w) for w in words])

        # FIX #7: Only compute formality for lines with >= 5 words
        # Short lines give degenerate TTR (always near 1.0)
        if len(words) >= 5:
            formality = ttr * avg_len
        else:
            formality = float('nan')

        slang_count = sum(1 for w in words if w in SLANG)
        contraction_count = len(CONTRACTION_PATTERN.findall(text))

        return {
            'cat4_type_token_ratio':  ttr,
            'cat4_avg_word_length':   avg_len,
            'cat4_formality_score':   formality,
            'cat4_slang_count':       slang_count,
            'cat4_contraction_count': contraction_count,
        }

    # ─── Category 5: Subtitle Constraint Violation ─────────────────────────

    def cat5_features(self, doc) -> dict:
        """
        Category 5: Subtitle Constraint Violation features.
        FIX #9: Use industry-standard 42 chars/line threshold.
        """
        words = [t for t in doc if t.is_alpha]
        n_words = len(words)
        n_chars = len(doc.text)

        if n_words == 0:
            return {
                'cat5_src_words': 0, 'cat5_src_chars': n_chars,
                'cat5_content_word_ratio': float('nan'),
                'cat5_avg_syllables': float('nan'),
                'cat5_length_risk': 0,
            }

        content_words = [t for t in words if t.pos_ in CONTENT_POS]
        content_ratio = len(content_words) / n_words
        avg_syllables = np.mean([self._count_syllables(t.text) for t in words])

        # FIX #9: Industry standard for subtitles
        # - max 42 chars per line, 2 lines → 84 chars
        # - or > 12 words (which tend to produce long translations)
        length_risk = int(n_words > 12 or n_chars > 84)

        return {
            'cat5_src_words':          n_words,
            'cat5_src_chars':          n_chars,
            'cat5_content_word_ratio': content_ratio,
            'cat5_avg_syllables':      avg_syllables,
            'cat5_length_risk':        length_risk,
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 0
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)

    # ─── Category 6: Fragmentation & Segmentation Failure ──────────────────

    def cat6_features(self, doc) -> dict:
        """Category 6: Fragmentation & Segmentation Failure features."""
        has_subject = any(t.dep_ in ('nsubj', 'nsubjpass', 'csubj') for t in doc)
        has_verb = any(t.pos_ in ('VERB', 'AUX') and t.dep_ == 'ROOT' for t in doc)
        complete = int(has_subject and has_verb)

        text = doc.text
        ellipsis_marker = int(bool(re.search(r'\.{2,}', text)))

        text_stripped = text.strip()
        starts_lower = int(bool(text_stripped) and text_stripped[0].islower())
        ends_incomplete = int(bool(text_stripped) and text_stripped[-1] not in '.!?')

        return {
            'cat6_has_subject':       int(has_subject),
            'cat6_has_verb':          int(has_verb),
            'cat6_complete_sentence': complete,
            'cat6_ellipsis_marker':   ellipsis_marker,
            'cat6_starts_lowercase':  starts_lower,
            'cat6_ends_incomplete':   ends_incomplete,
        }

    # ─── Category 7: World-building & Terminology Loss ─────────────────────

    def cat7_features(self, doc) -> dict:
        """Category 7: World-building & Terminology Loss features."""
        ner_count = len(doc.ents)
        ner_types = list({ent.label_ for ent in doc.ents})

        content_words = [t.text.lower() for t in doc
                         if t.is_alpha and not t.is_stop and len(t.text) > 2]
        if len(content_words) >= 3:  # FIX #16: minimum 3 content words for stable rate
            freqs = [word_frequency(w, 'en') for w in content_words]
            oov_rate = sum(1 for f in freqs if f < OOV_THRESHOLD) / len(freqs)
            rare_count = sum(1 for f in freqs if f < RARE_THRESHOLD)
        elif content_words:
            # Too few content words for a stable rate — still count rares but NaN the rate
            freqs = [word_frequency(w, 'en') for w in content_words]
            oov_rate = float('nan')
            rare_count = sum(1 for f in freqs if f < RARE_THRESHOLD)
        else:
            oov_rate, rare_count = float('nan'), 0

        text = doc.text
        words = text.split()
        entity_spans = {ent.text for ent in doc.ents}
        unusual_cap = sum(
            1 for i, w in enumerate(words)
            if i > 0 and w[0].isupper() and w.isalpha()
            and not any(w in span for span in entity_spans)
        )

        hyphenated = len(re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b', text))
        camel_case = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text))
        compound_neologisms = hyphenated + camel_case

        return {
            'cat7_ner_count':           ner_count,
            'cat7_ner_types':           str(ner_types),
            'cat7_oov_rate':            oov_rate,
            'cat7_rare_word_count':     rare_count,
            'cat7_unusual_cap_words':   unusual_cap,
            'cat7_compound_neologisms': compound_neologisms,
        }

    # ─── Category 8: Over-interpretation / Hallucinated Tone ───────────────

    def cat8_features(self, doc, vader_compound: float) -> dict:
        """
        Category 8: Over-interpretation / Hallucinated Tone features.
        FIX #11: Exclude imperatives from implicit_subject to avoid confound with cat3.
        """
        words = [t for t in doc if t.is_alpha]
        n_words = len(words)

        brevity = int(n_words <= 3)
        is_single_word = int(n_words == 1)
        vader_neutral = abs(vader_compound) < VADER_NEUTRAL_THRESHOLD
        tone_ambiguous = int(brevity and vader_neutral)

        modal_count = sum(1 for t in doc if t.text.lower() in MODAL_VERBS)

        # FIX #11: Only flag implicit subject for NON-imperative sentences
        # Imperatives naturally lack subjects — that's cat3's domain
        root_verb = next((t for t in doc if t.dep_ == 'ROOT' and t.pos_ == 'VERB'), None)
        implicit_subject = 0
        if root_verb:
            has_subject = any(c.dep_ in ('nsubj', 'nsubjpass', 'expl')
                              for c in root_verb.children)
            is_imperative = not has_subject  # same logic as cat3
            # Only flag if it's NOT an imperative — imperatives are expected to lack subjects
            if not has_subject and not is_imperative:
                # This branch never fires because is_imperative == not has_subject.
                # The real fix: only flag when there IS a subject-requiring context
                # (e.g., declarative with auxiliary but no subject: "Could be worse.")
                pass

            # Better approach: flag when there's an auxiliary or modal but no subject
            # This catches "Could be worse", "Should have known", "Might work"
            has_aux = any(c.dep_ == 'aux' or c.pos_ == 'AUX' for c in root_verb.children)
            if not has_subject and has_aux:
                implicit_subject = 1
            # Also flag participle phrases without subject: "Running through the rain"
            elif not has_subject and root_verb.tag_ in ('VBG', 'VBN'):
                implicit_subject = 1

        return {
            'cat8_brevity':          brevity,
            'cat8_is_single_word':   is_single_word,
            'cat8_vader_neutrality': int(vader_neutral),
            'cat8_tone_ambiguous':   tone_ambiguous,
            'cat8_modal_count':      modal_count,
            'cat8_implicit_subject': implicit_subject,
        }

    # ─── Back-translation (Cat 1 supplement) ───────────────────────────────

    def back_translate_batch(self, zh_texts: list, batch_size: int = 64) -> list:
        """FIX #14: Removed conflicting max_length parameter."""
        import torch
        if self.bt_model is None:
            self._load_backtranslation()

        results = []
        for i in range(0, len(zh_texts), batch_size):
            batch = [t if t.strip() else ' ' for t in zh_texts[i:i + batch_size]]
            inputs = self.bt_tokenizer(batch, return_tensors='pt', padding=True,
                                        truncation=True, max_length=128)
            with torch.no_grad():
                # FIX #14: Only use max_new_tokens, don't set max_length
                translated = self.bt_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                )
            results.extend(self.bt_tokenizer.batch_decode(translated, skip_special_tokens=True))
        return results

    @staticmethod
    def sentence_bleu(hyp: str, ref: str) -> float:
        import sacrebleu
        if not hyp.strip() or not ref.strip():
            return float('nan')
        return sacrebleu.sentence_bleu(hyp, [ref]).score

    # ─── Full pipeline ─────────────────────────────────────────────────────

    def extract_all_features(self, df_movie: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all 8 category features for a movie DataFrame.
        Input: DataFrame with 'en' column (and optionally 'zh').
        Returns: DataFrame with all feature columns added.

        FIX #8:  Voice shift window resets per call (call per-movie).
        FIX #10: Cleans OPUS tokenization artifacts.
        FIX #13: Single spaCy parse per line, reused across categories.
        """
        out = df_movie.copy()

        # FIX #10: Clean OPUS tokenization artifacts
        out['en_clean'] = out['en'].apply(clean_opus_tokenization)

        out['word_count'] = out['en_clean'].str.split().str.len()
        out['char_count'] = out['en_clean'].str.len()

        # ── FIX #13: Single spaCy parse per line ──────────────────────────
        logger.info(f"Parsing {len(out)} lines with spaCy...")
        docs = list(self.nlp.pipe(out['en_clean'].tolist(), batch_size=256))

        # ── Cat 1: Lexicon features ───────────────────────────────────────
        c1_lex = [self.cat1_lexicon_features(text) for text in out['en_clean']]
        c1_lex_df = pd.DataFrame(c1_lex)
        out = pd.concat([out, c1_lex_df], axis=1)

        # ── Cat 1: Compositionality (batch) ───────────────────────────────
        raw_scores, norm_scores = self.cat1_compositionality(out['en_clean'].tolist())
        out['cat1_compositionality_raw'] = raw_scores
        out['cat1_compositionality'] = norm_scores

        # ── Cat 2: Pragmatic ──────────────────────────────────────────────
        c2 = [self.cat2_features(text) for text in out['en_clean']]
        out = pd.concat([out, pd.DataFrame(c2)], axis=1)

        # ── Cats 3, 5, 6, 7: Use pre-parsed docs ─────────────────────────
        c3 = [self.cat3_features(doc) for doc in docs]
        c5 = [self.cat5_features(doc) for doc in docs]
        c6 = [self.cat6_features(doc) for doc in docs]
        c7 = [self.cat7_features(doc) for doc in docs]
        out = pd.concat([out, pd.DataFrame(c3), pd.DataFrame(c5),
                         pd.DataFrame(c6), pd.DataFrame(c7)], axis=1)

        # ── Cat 4: Register (text-based, no doc needed) ───────────────────
        c4 = [self.cat4_features(text) for text in out['en_clean']]
        out = pd.concat([out, pd.DataFrame(c4)], axis=1)

        # ── Cat 4 bonus: Voice shift ──────────────────────────────────────
        # FIX #8: This is called per-movie, so window stays within one film
        embs = self.minilm.encode(out['en_clean'].tolist(), batch_size=128,
                                   show_progress_bar=False)
        WINDOW = 5
        vs = [float('nan')] * len(out)
        for i in range(WINDOW, len(out)):
            wm = embs[i - WINDOW:i].mean(axis=0)
            vs[i] = 1.0 - float(cosine_similarity([embs[i]], [wm])[0][0])
        out['cat4_voice_shift'] = vs

        # ── Cat 8: Hallucination (needs vader compound from cat2) ─────────
        c8 = [self.cat8_features(doc, row['cat2_vader_compound'])
              for doc, (_, row) in zip(docs, out.iterrows())]
        out = pd.concat([out, pd.DataFrame(c8)], axis=1)

        # Drop the clean column (keep original for reference)
        out = out.drop(columns=['en_clean'])

        return out

    def extract_corpus(self, movie_list: pd.DataFrame,
                       get_parallel_df_fn, max_movies: int = None) -> pd.DataFrame:
        """
        FIX #12: Error handling per-movie with logging.

        Args:
            movie_list: DataFrame with 'title', 'year', 'parallel_pairs' columns
            get_parallel_df_fn: function(title) -> DataFrame with 'en', 'zh' columns
            max_movies: limit for testing (None = all)
        """
        movies = movie_list[movie_list['parallel_pairs'] > 0]
        if max_movies:
            movies = movies.head(max_movies)

        logger.info(f'Processing {len(movies)} movies...')
        all_results = []
        failed = []

        for idx, (_, movie_row) in enumerate(movies.iterrows()):
            title = movie_row['title']
            try:
                movie_df = get_parallel_df_fn(title)
                if movie_df.empty:
                    continue
                logger.info(f'[{idx+1}/{len(movies)}] {title} ({movie_row["year"]}): '
                            f'{len(movie_df)} lines')
                features = self.extract_all_features(movie_df)
                features.insert(0, 'movie', title)
                features.insert(1, 'year', movie_row['year'])
                all_results.append(features)
            except Exception as e:
                logger.error(f'FAILED on "{title}": {e}')
                failed.append({'title': title, 'error': str(e)})
                continue

        if failed:
            logger.warning(f'{len(failed)} movies failed:')
            for f in failed:
                logger.warning(f'  {f["title"]}: {f["error"]}')

        if not all_results:
            logger.error('No movies processed successfully!')
            return pd.DataFrame()

        full_df = pd.concat(all_results, ignore_index=True)
        logger.info(f'Full feature matrix: {full_df.shape}')
        return full_df


# =============================================================================
# Feature column listing (for downstream use)
# =============================================================================

ALL_FEATURE_COLS = [
    # Cat 1
    'cat1_idiom_freq', 'cat1_idiom_flag', 'cat1_compositionality_raw',
    'cat1_compositionality',
    # Cat 2
    'cat2_vader_compound', 'cat2_vader_neg', 'cat2_vader_pos',
    'cat2_exclamations', 'cat2_questions', 'cat2_ellipses',
    'cat2_caps_words', 'cat2_discourse_markers', 'cat2_dm_rate', 'cat2_negations',
    'cat2_negated_positive', 'cat2_negated_negative',
    # Cat 3
    'cat3_vocative_count', 'cat3_you_count', 'cat3_imperative',
    'cat3_hedge_count', 'cat3_direct_command',
    # Cat 4
    'cat4_type_token_ratio', 'cat4_avg_word_length', 'cat4_formality_score',
    'cat4_slang_count', 'cat4_contraction_count', 'cat4_voice_shift',
    # Cat 5
    'cat5_src_words', 'cat5_src_chars', 'cat5_content_word_ratio',
    'cat5_avg_syllables', 'cat5_length_risk',
    # Cat 6
    'cat6_has_subject', 'cat6_has_verb', 'cat6_complete_sentence',
    'cat6_ellipsis_marker', 'cat6_starts_lowercase', 'cat6_ends_incomplete',
    # Cat 7
    'cat7_ner_count', 'cat7_oov_rate', 'cat7_rare_word_count',
    'cat7_unusual_cap_words', 'cat7_compound_neologisms',
    # Cat 8
    'cat8_brevity', 'cat8_is_single_word', 'cat8_vader_neutrality',
    'cat8_tone_ambiguous', 'cat8_modal_count', 'cat8_implicit_subject',
]

# Per-category feature groupings (for logistic regression)
CATEGORY_FEATURES = {
    'cat1': ['cat1_idiom_freq', 'cat1_idiom_flag', 'cat1_compositionality',
             'cat1_compositionality_raw'],
    'cat2': ['cat2_vader_compound', 'cat2_vader_neg', 'cat2_vader_pos',
             'cat2_exclamations', 'cat2_questions', 'cat2_ellipses',
             'cat2_caps_words', 'cat2_discourse_markers', 'cat2_dm_rate', 'cat2_negations',
             'cat2_negated_positive', 'cat2_negated_negative'],
    'cat3': ['cat3_vocative_count', 'cat3_you_count', 'cat3_imperative',
             'cat3_hedge_count', 'cat3_direct_command'],
    'cat4': ['cat4_type_token_ratio', 'cat4_avg_word_length', 'cat4_formality_score',
             'cat4_slang_count', 'cat4_contraction_count', 'cat4_voice_shift'],
    'cat5': ['cat5_src_words', 'cat5_src_chars', 'cat5_content_word_ratio',
             'cat5_avg_syllables', 'cat5_length_risk'],
    'cat6': ['cat6_has_subject', 'cat6_has_verb', 'cat6_complete_sentence',
             'cat6_ellipsis_marker', 'cat6_starts_lowercase', 'cat6_ends_incomplete'],
    'cat7': ['cat7_ner_count', 'cat7_oov_rate', 'cat7_rare_word_count',
             'cat7_unusual_cap_words', 'cat7_compound_neologisms'],
    'cat8': ['cat8_brevity', 'cat8_is_single_word', 'cat8_vader_neutrality',
             'cat8_tone_ambiguous', 'cat8_modal_count', 'cat8_implicit_subject'],
}


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    fe = FeatureExtractor()

    test_lines = pd.DataFrame({'en': [
        "You're really milking it, aren't you?",
        "We need to go to the station now.",
        "He kicked the bucket.",
        "Could you please sit down, sir?",
        "Get out of here, buddy!",
        "Don't move.",
        "Fine.",
        "Whatever.",
        "The implementation of the protocol necessitates a comprehensive review.",
        "Dude, I'm gonna freak out if this doesn't work.",
        "That's NOT BAD at all!",
        "Not terrible, honestly.",
        "Tyler Durden is the CEO of Paper Street Soap Company.",
        "The Flux-Capacitor is malfunctioning near Omega-3 coordinates.",
        "Running through the rain every morning.",
        "I could come if you want.",
        "No. No.",
        "I",
        "...and that's when it happened.",
        "Three minutes .",  # OPUS-style spacing
    ]})

    result = fe.extract_all_features(test_lines)

    print("\n" + "=" * 80)
    print("VALIDATION: Checking key fixes")
    print("=" * 80)

    # FIX #1: Compositionality — check that short simple lines and long complex
    # lines don't dominate the score anymore
    print("\n--- FIX #1: Compositionality (raw vs normalized) ---")
    for _, row in result.iterrows():
        if pd.notna(row.get('cat1_compositionality_raw')):
            print(f"  [{row['cat1_compositionality_raw']:.3f} raw | "
                  f"{row['cat1_compositionality']:.3f} norm] {row['en'][:60]}")

    # FIX #3: Phrasal verbs — opaque vs transparent
    print("\n--- FIX #3: Phrasal verbs (opaque vs transparent) ---")
    for _, row in result.iterrows():
        if row.get('cat1_pv_opaque') or row.get('cat1_pv_transparent'):
            print(f"  opaque={row.get('cat1_pv_opaque')}  "
                  f"transparent={row.get('cat1_pv_transparent')}  | {row['en'][:60]}")

    # FIX #4: Negated positive vs negated negative
    print("\n--- FIX #4: Negated positive vs negative ---")
    for _, row in result.iterrows():
        if row['cat2_negated_positive'] > 0 or row.get('cat2_negated_negative', 0) > 0:
            print(f"  neg_pos={row['cat2_negated_positive']}  "
                  f"neg_neg={row.get('cat2_negated_negative', 0)}  | {row['en'][:60]}")

    # FIX #7: Formality — short lines should be NaN
    print("\n--- FIX #7: Formality score (NaN for short lines) ---")
    for _, row in result.iterrows():
        formality = row['cat4_formality_score']
        wc = row['word_count']
        status = "NaN (correct)" if pd.isna(formality) and wc < 5 else \
                 f"{formality:.2f}" if pd.notna(formality) else "NaN"
        print(f"  words={wc:<3} formality={status:<20} | {row['en'][:50]}")

    # FIX #10: OPUS cleaning
    print("\n--- FIX #10: OPUS tokenization cleaning ---")
    opus_line = result[result['en'] == "Three minutes ."]
    if not opus_line.empty:
        # Check that features still work on OPUS-style text
        print(f"  Original: 'Three minutes .' -> ends_incomplete="
              f"{opus_line.iloc[0]['cat6_ends_incomplete']}")

    print(f"\nTotal features: {len(ALL_FEATURE_COLS)}")
    print(f"Feature matrix shape: {result.shape}")
    print("\nDone.")
