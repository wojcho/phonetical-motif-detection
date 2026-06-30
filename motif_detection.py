from phonemizer import phonemize
import re
import csv
from dataclasses import dataclass
import numpy as np
import stumpy

def text_to_ipa(text: str, lang: str = "pl") -> str:
    """
    Convert text to IPA using espeak backend through phonemizer
    """
    ipa = phonemize(
        text,
        language=lang,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
        njobs=1
    )
    print(ipa)
    return ipa

def load_feature_weights(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items() if v not in ("", None)}

FEATURE_WEIGHTS = load_feature_weights("./feature_weights.csv")

CSV_FIELD_MAP = {
    "syl": "syllabic",
    "son": "sonorant",
    "cons": "consonant",
    "cont": "continuant",
    "delrel": "delayed_release",
    "lat": "lateral",
    "nas": "nasal",
    "strid": "strident",
    "voi": "voiced",
    "sg": "spread_glottis",
    "cg": "constricted_glottis",
    "ant": "anterior_place",
    "cor": "coronal_place",
    "distr": "distributed",
    "lab": "labial",
    "hi": "high_vowel_height",
    "lo": "low_vowel_height",
    "back": "back_vowel",
    "round": "rounded",
    "velaric": "velaric",
    "tense": "tense",
    "long": "long",
    "hitone": "high_tone",
    "hireg": "high_register",
}

_PHONEME_CACHE: dict[tuple[str, bool], Phoneme] = {}

@dataclass(frozen=True, slots=True)
class Phoneme:
    symbol: str
    stress: bool = False

    syllabic: bool | None = None
    sonorant: bool | None = None
    consonant: bool | None = None
    continuant: bool | None = None
    delayed_release: bool | None = None
    lateral: bool | None = None
    nasal: bool | None = None
    strident: bool | None = None
    voiced: bool | None = None
    spread_glottis: bool | None = None
    constricted_glottis: bool | None = None
    anterior_place: bool | None = None
    coronal_place: bool | None = None
    distributed: bool | None = None
    labial: bool | None = None
    high_vowel_height: bool | None = None
    low_vowel_height: bool | None = None
    back_vowel: bool | None = None
    rounded: bool | None = None
    velaric: bool | None = None
    tense: bool | None = None
    long: bool | None = None
    high_tone: bool | None = None
    high_register: bool | None = None

    @staticmethod
    def from_symbol(symbol: str, stress: bool) -> Phoneme:
        row = IPA_FEATURES.get(symbol)
        if row is None:
            raise KeyError(f"Symbol {symbol!r} not found in CSV")

        def parse_cell(v: str) -> bool | None:
            v = (v or "").strip()
            if v in ("+", "True", "true"):
                return True
            if v in ("-", "False", "false"):
                return False
            if v in ("0", ""):
                return None
            raise ValueError(f"Unrecognized feature cell: {v!r} for symbol {symbol!r}")

        p = Phoneme(symbol=symbol, stress=stress)

        kwargs: dict[str, bool | None] = {}
        for csv_col, attr_name in CSV_FIELD_MAP.items():
            if csv_col in row:
                kwargs[attr_name] = parse_cell(row[csv_col])

        return Phoneme(symbol=symbol, stress=stress, **kwargs)

    def __str__(self: Phoneme) -> str:
        return ("ˈ" if self.stress else "") + self.symbol

    def __repr__(self: Phoneme) -> str:
        return self.__str__()

def load_ipa_features(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load Panphon phoneme features from CSV file
    """
    feats: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row["ipa"].strip()
            feats[sym] = row
    return feats

IPA_FEATURES = load_ipa_features("./ipa_all.csv")

def get_phoneme(symbol: str, stress: bool) -> Phoneme:
    """
    For flyweight pattern, to not duplicate Phonemes in memory
    """
    key = (symbol, stress)
    p = _PHONEME_CACHE.get(key)
    if p is None:
        p = Phoneme.from_symbol(symbol, stress=stress)
        _PHONEME_CACHE[key] = p
    return p

AFFRICATES = ["t͡s", "d͡z", "t͡ʂ", "d͡ʐ", "t͡ʃ", "t͡ɕ"]
PALATAL_MARKER = "ʲ"

AFFRICATE_NORMALIZATION = [
    (r"t\s?͡?\s?s", "t͡s"),
    (r"d\s?͡?\s?z", "d͡z"),
    (r"t\s?͡?\s?ʂ", "t͡ʂ"),
    (r"d\s?͡?\s?ʐ", "d͡ʐ"),
    (r"t\s?͡?\s?ʃ", "t͡ʃ"),
    (r"t\s?͡?\s?ɕ", "t͡ɕ"),
]

def normalize_ipa(ipa: str) -> str:
    """
    Make representation of various phonemes more uniform, remove low signal data
    """
    for pattern, repl in AFFRICATE_NORMALIZATION:
        ipa = re.sub(pattern, repl, ipa)
    ipa = re.sub("ˌ", "", ipa) # Remove secondary stress
    return ipa

def ipa_to_segments(ipa: str):
    """
    Tokenize IPA to Phoneme data classes
    """
    ipa = normalize_ipa(ipa)

    segments = []
    i = 0
    stress = False

    while i < len(ipa):

        # stress marker
        if ipa[i] == "ˈ":
            stress = True
            i += 1
            continue

        # whitespace = syllable boundary reset stress (optional design choice)
        if ipa[i].isspace():
            i += 1
            continue

        # affricates
        matched = False
        for aff in AFFRICATES:
            if ipa.startswith(aff, i):
                segments.append(get_phoneme(aff, stress=stress))
                stress = False
                i += len(aff)
                matched = True
                break
        if matched:
            continue

        # palatalized consonants (simple heuristic)
        if i + 1 < len(ipa) and ipa[i+1] == PALATAL_MARKER:
            segments.append(get_phoneme(ipa[i] + "ʲ", stress=stress))
            stress = False
            i += 2
            continue

        # normal phoneme
        segments.append(get_phoneme(ipa[i], stress=stress))
        stress = False
        i += 1

    return segments

def phoneme_to_vec(p: Phoneme, weights: dict[str, float]) -> np.array:
    """
    Convert Phoneme to vector np.array representation
    using feature order from CSV_FIELD_MAP keys
    """
    # Use CSV_FIELD_MAP order (source CSV columns), mapped to Phoneme attribute names.
    ordered_fields = []
    for csv_key, ph_attr in CSV_FIELD_MAP.items():
        if csv_key in weights:
            ordered_fields.append((csv_key, ph_attr, weights[csv_key]))

    vec = np.zeros(len(ordered_fields), dtype=np.float32)

    for j, (csv_key, ph_attr, _) in enumerate(ordered_fields):
        val = getattr(p, ph_attr, None)
        w = weights[csv_key]
        if val is True:
            vec[j] = w
        elif val is False:
            vec[j] = -w
        else:
            vec[j] = 0.0
    
    return vec

if __name__ == "__main__":
    sample = "więcej przędzej przestrzeń wrzenie tego nigdy te gonitwy stracenia znaczenia powietrze po jeszcze"
    ipa_phonemes = ipa_to_segments(text_to_ipa(sample))
    print(ipa_phonemes)
    vectors_representation = np.array([phoneme_to_vec(phoneme, FEATURE_WEIGHTS) for phoneme in ipa_phonemes], dtype=np.float64)
    print(vectors_representation)
    vectors_for_stumpy = vectors_representation.T
    m = 4 # TODO later using for loop with various m, and keep greedily longest pattern found rather than its subsets
    P, I = stumpy.mstump(vectors_for_stumpy, m=m) # P.shape == (num_features, n_windows), I.shape == (num_features, n_windows)
    profile = P[-1]
    indices = I[-1]
    best = np.argsort(profile)[:20]
    for i in best:
        j = indices[i]
        print(
            i,
            j,
            profile[i],
            ipa_phonemes[i:i+m],
            ipa_phonemes[j:j+m],
        )
    motif_distances, motif_indices, motif_neighbors, motif_subspaces = (
        stumpy.mmotifs(vectors_for_stumpy, P, I)
    )
    print(motif_distances, motif_indices, motif_neighbors, motif_subspaces)

# `[vʲ, ˈɛ, n, t͡s, ɛ, j, p, ʃ, ˈɛ, n, d, ʑ, ɛ, j, p, ʃ, ˈɛ, s, t͡ʃ, ɛ, ɲ, v, ʒ, ˈɛ, ɲʲ, ɛ, t, ɛ, ɡ, ɔ, ɲ, ˈi, ɡ, d, ɨ, t, ɛ, ɡ, ɔ, ɲ, ˈi, t, f, ɨ, s, t, r, a, t͡s, ˈɛ, ɲʲ, a, z, n, a, t͡ʃ, ˈɛ, ɲʲ, a, p, ɔ, vʲ, ˈɛ, t͡ʃ, ɛ, p, ɔ, j, ˈɛ, ʃ, t͡ʃ, ɛ]`
# ```
# [[-1.    -1.     1.    ... -0.25   0.    -0.125]
#  [ 1.     1.    -1.    ... -0.25  -0.25  -0.125]
#  [-1.     1.     1.    ... -0.25   0.    -0.125]
#  ...
#  [-1.    -1.     1.    ... -0.25   0.    -0.125]
#  [-1.    -1.     1.    ... -0.25   0.    -0.125]
#  [ 1.     1.    -1.    ... -0.25  -0.25  -0.125]]
# ```
# Using these words and their extracted features, and feature weights in matrix representation, how could repeating motifs be detected?
# For example, these words all have components which are phonetically similar "więcej - przędzej - przestrzeń - wrzenie", or "tego nigdy - te gonitwy"
# These similarities are at smaller level than of entire words

# 5 13 0.0 [j, p, ʃ, ˈɛ] [j, p, ʃ, ˈɛ]
# 4 12 0.0 [ɛ, j, p, ʃ] [ɛ, j, p, ʃ]
# 12 4 0.0 [ɛ, j, p, ʃ] [ɛ, j, p, ʃ]
# 13 5 0.0 [j, p, ʃ, ˈɛ] [j, p, ʃ, ˈɛ]
# 27 36 0.0 [ɛ, ɡ, ɔ, ɲ] [ɛ, ɡ, ɔ, ɲ]
# 36 27 0.0 [ɛ, ɡ, ɔ, ɲ] [ɛ, ɡ, ɔ, ɲ]
# 28 37 1.9157658414042743e-09 [ɡ, ɔ, ɲ, ˈi] [ɡ, ɔ, ɲ, ˈi]
# 26 35 1.9157658414042743e-09 [t, ɛ, ɡ, ɔ] [t, ɛ, ɡ, ɔ]
# 35 26 1.9157658414042743e-09 [t, ɛ, ɡ, ɔ] [t, ɛ, ɡ, ɔ]
# 37 28 1.9157658414042743e-09 [ɡ, ɔ, ɲ, ˈi] [ɡ, ɔ, ɲ, ˈi]
# 16 68 0.25712973861329 [ˈɛ, s, t͡ʃ, ɛ] [ˈɛ, ʃ, t͡ʃ, ɛ]
# 68 16 0.25712973861329 [ˈɛ, ʃ, t͡ʃ, ɛ] [ˈɛ, s, t͡ʃ, ɛ]
# 34 25 0.295728814270041 [ɨ, t, ɛ, ɡ] [ɛ, t, ɛ, ɡ]
# 25 34 0.295728814270041 [ɛ, t, ɛ, ɡ] [ɨ, t, ɛ, ɡ]
# 54 47 0.3103830530405927 [a, t͡ʃ, ˈɛ, ɲʲ] [a, t͡s, ˈɛ, ɲʲ]
# 55 48 0.3103830530405927 [t͡ʃ, ˈɛ, ɲʲ, a] [t͡s, ˈɛ, ɲʲ, a]
# 48 55 0.3103830530405927 [t͡s, ˈɛ, ɲʲ, a] [t͡ʃ, ˈɛ, ɲʲ, a]
# 47 54 0.3103830530405927 [a, t͡s, ˈɛ, ɲʲ] [a, t͡ʃ, ˈɛ, ɲʲ]
# 38 23 0.39396502456440774 [ɔ, ɲ, ˈi, t] [ˈɛ, ɲʲ, ɛ, t]
# 23 38 0.39396502456440774 [ˈɛ, ɲʲ, ɛ, t] [ɔ, ɲ, ˈi, t]

# TODO
# Change Panphon CSV to have aligned names
# Support patterns of various lengths pairwise same, and join by greedily leaving longest one and removing those which are its subsets
# Original text of pairs would be found and motifs visualized, it could be done by retaining indices of occurence in original phonemes, refactored to separate PhonemeOccurence with stress and location, and as composition holding flyweight PhonemeType

# Other more expensive algorithms could be ran on each window to reveal which have least distance in respect to some distance measure
# Support patterns which have different lengths because of some insertions or deletions - Needleman–Wunsch or Smith–Waterman but not with equality but with feature vectors, Dynamic Time Warping, Soft-DTW
# Support patterns which have a few swaps - Damerau-style alignment, elastic matching
# On these generated pairs, some unified scoring would be used for final score of pairs, it could be expensive because there should not be that many of them
