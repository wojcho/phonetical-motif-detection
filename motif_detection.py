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

        kwargs: dict[str, bool | None] = {}
        for attr_name in Phoneme.__dataclass_fields__.keys():
            if attr_name in ("symbol", "stress"):
                continue
            if attr_name in row:
                kwargs[attr_name] = parse_cell(row[attr_name])

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
            sym = row["symbol"].strip()
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
    using feature order from weights.csv column names
    """
    ordered_fields = list(weights.keys()) # using order from weights CSV

    vec = np.zeros(len(ordered_fields), dtype=np.float32)

    for j, feature_key in enumerate(ordered_fields):
        val = getattr(p, feature_key, None)
        w = weights[feature_key]
        if val is True:
            vec[j] = w
        elif val is False:
            vec[j] = -w
        else:
            vec[j] = 0.0
    
    return vec

@dataclass(frozen=True)
class MotifSpan:
    start: int
    end: int # exclusive
    match_start: int
    match_end: int # exclusive
    distance: float
    m: int

    @property
    def interval(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def match_interval(self) -> Tuple[int, int]:
        return (self.match_start, self.match_end)

    def covers(self, other: "MotifSpan") -> bool:
        return self.start <= other.start and self.end >= other.end and self.match_start <= other.match_start and self.match_end >= other.match_end

def extract_pairwise_motifs(X: np.ndarray, m_values: Iterable=range(2, 9), top_k_per_window: int=20):
    """
    X: shape (n_features, n_timestamps) for stumpy
    Returns motif candidates across multiple window lengths.
    """
    candidates: list[MotifSpan] = []

    for m in m_values:
        P, I = stumpy.mstump(X, m=m)
        profile = P[-1]
        indices = I[-1]

        best = np.argsort(profile)[:top_k_per_window]
        for i in best:
            j = int(indices[i])
            if j < 0:
                continue
            candidates.append(
                MotifSpan(
                    start=int(i),
                    end=int(i + m),
                    match_start=j,
                    match_end=int(j + m),
                    distance=float(profile[i]),
                    m=m,
                )
            )

    return candidates

def greedy_keep_longest_non_subset(candidates: List[MotifSpan]) -> List[MotifSpan]:
    """
    Remove candidates which are subsets of longer spans
    This also takes distance scores of pairs into account
    If adding more phonemes does not cause match to be worse then it is kept
    """
    candidates = sorted(
        candidates,
        key=lambda c: (round(c.distance, 5), -c.m, c.start, c.match_start) # Without rounding there is floating point noise in distance scores
    )

    kept: List[MotifSpan] = []

    def is_subset_of_kept(c: MotifSpan) -> bool:
        for k in kept:
            if k.covers(c):
                return True
            if k.covers(MotifSpan(c.match_start, c.match_end, c.start, c.end, c.distance, c.m)):
                return True
        return False

    for c in candidates:
        if not is_subset_of_kept(c):
            kept.append(c)

    return kept

if __name__ == "__main__":
    sample = "nie chcę, nie chcę"
    ipa_phonemes = ipa_to_segments(text_to_ipa(sample))
    print(ipa_phonemes)
    vectors_representation = np.array([phoneme_to_vec(phoneme, FEATURE_WEIGHTS) for phoneme in ipa_phonemes], dtype=np.float64)
    print(vectors_representation)
    vectors_for_stumpy = vectors_representation.T
    X = vectors_representation.T
    cands = extract_pairwise_motifs(X, m_values=range(3, 9), top_k_per_window=30)
    kept = greedy_keep_longest_non_subset(cands)
    for c in kept:
        print(c.distance, c.m, ipa_phonemes[c.start:c.end], ipa_phonemes[c.match_start:c.match_end])

# ɲʲˈɛxtsɛ ɲʲˈɛxtsɛ
# [ɲʲ, ˈɛ, x, t͡s, ɛ, ɲʲ, ˈɛ, x, t͡s, ɛ]
# [[-1.     1.     1.    -0.5   -0.25  -0.25   0.25  -0.125  0.125 -0.125
#   -0.125 -0.25  -0.25   0.    -0.25   0.25  -0.25  -0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [ 1.     1.    -1.     0.5   -0.25  -0.25  -0.25  -0.125  0.125 -0.125
#   -0.125  0.    -0.25   0.    -0.25  -0.25  -0.25  -0.25  -0.25  -0.25
#   -0.125 -0.25 ]
#  [-1.    -1.     1.     0.5   -0.25  -0.25  -0.25  -0.125 -0.125 -0.125
#   -0.125 -0.25  -0.25   0.    -0.25   0.25  -0.25   0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [-1.    -1.     1.    -0.5    0.25  -0.25  -0.25   0.125 -0.125 -0.125
#   -0.125  0.25   0.25  -0.125 -0.25  -0.25  -0.25  -0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [ 1.     1.    -1.     0.5   -0.25  -0.25  -0.25  -0.125  0.125 -0.125
#   -0.125  0.    -0.25   0.    -0.25  -0.25  -0.25  -0.25  -0.25  -0.25
#   -0.125 -0.25 ]
#  [-1.     1.     1.    -0.5   -0.25  -0.25   0.25  -0.125  0.125 -0.125
#   -0.125 -0.25  -0.25   0.    -0.25   0.25  -0.25  -0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [ 1.     1.    -1.     0.5   -0.25  -0.25  -0.25  -0.125  0.125 -0.125
#   -0.125  0.    -0.25   0.    -0.25  -0.25  -0.25  -0.25  -0.25  -0.25
#   -0.125 -0.25 ]
#  [-1.    -1.     1.     0.5   -0.25  -0.25  -0.25  -0.125 -0.125 -0.125
#   -0.125 -0.25  -0.25   0.    -0.25   0.25  -0.25   0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [-1.    -1.     1.    -0.5    0.25  -0.25  -0.25   0.125 -0.125 -0.125
#   -0.125  0.25   0.25  -0.125 -0.25  -0.25  -0.25  -0.25  -0.25   0.
#   -0.125 -0.25 ]
#  [ 1.     1.    -1.     0.5   -0.25  -0.25  -0.25  -0.125  0.125 -0.125
#   -0.125  0.    -0.25   0.    -0.25  -0.25  -0.25  -0.25  -0.25  -0.25
#   -0.125 -0.25 ]]
# 5.652362176780671e-09 5 [ɲʲ, ˈɛ, x, t͡s, ɛ] [ɲʲ, ˈɛ, x, t͡s, ɛ]
# 1.0431424707011683 3 [ɛ, ɲʲ, ˈɛ] [ˈɛ, x, t͡s]
# 1.0715384270802417 3 [t͡s, ɛ, ɲʲ] [ɲʲ, ˈɛ, x]
# 1.5930445947779475 4 [x, t͡s, ɛ, ɲʲ] [ɛ, ɲʲ, ˈɛ, x]
# 1.747572274270889 4 [t͡s, ɛ, ɲʲ, ˈɛ] [ɲʲ, ˈɛ, x, t͡s]
# 2.1402361058251915 5 [ˈɛ, x, t͡s, ɛ, ɲʲ] [ɛ, ɲʲ, ˈɛ, x, t͡s]
# 2.1402361058251915 5 [x, t͡s, ɛ, ɲʲ, ˈɛ] [ɲʲ, ˈɛ, x, t͡s, ɛ]
# 2.1402361058251915 5 [t͡s, ɛ, ɲʲ, ˈɛ, x] [ɲʲ, ˈɛ, x, t͡s, ɛ]
# 2.24106786620025 6 [ˈɛ, x, t͡s, ɛ, ɲʲ, ˈɛ] [ɛ, ɲʲ, ˈɛ, x, t͡s, ɛ]
# 2.360021097054136 6 [ɲʲ, ˈɛ, x, t͡s, ɛ, ɲʲ] [t͡s, ɛ, ɲʲ, ˈɛ, x, t͡s]
# 2.4341212550852145 7 [ɲʲ, ˈɛ, x, t͡s, ɛ, ɲʲ, ˈɛ] [t͡s, ɛ, ɲʲ, ˈɛ, x, t͡s, ɛ]

# TODO
# Original text of pairs would be found and motifs visualized, it could be done by retaining indices of occurence in original phonemes, refactored to separate PhonemeOccurence with stress and location, and as composition holding flyweight PhonemeType
# Improve using a measurement rather than fixed amount of top pairs

# Other more expensive algorithms could be ran on each window to reveal which have least distance in respect to some distance measure
# Support patterns which have different lengths because of some insertions or deletions - Needleman–Wunsch or Smith–Waterman but not with equality but with feature vectors, Dynamic Time Warping, Soft-DTW
# Support patterns which have a few swaps - Damerau-style alignment, elastic matching
# On these generated pairs, some unified scoring would be used for final score of pairs, it could be expensive because there should not be that many of them
