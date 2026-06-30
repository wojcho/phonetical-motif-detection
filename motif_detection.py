from phonemizer import phonemize
import re
import csv
from dataclasses import dataclass
import numpy as np
import stumpy
import unicodedata
from functools import lru_cache

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

    def as_vector(self: Phoneme, weights: dict[str, float] = FEATURE_WEIGHTS) -> np.array:
      """
      Convert Phoneme to vector np.array representation
      using feature order from weights.csv column names
      """
      ordered_fields = list(weights.keys()) # using order from weights CSV

      vec = np.zeros(len(ordered_fields), dtype=np.float64)

      for j, feature_key in enumerate(ordered_fields):
          val = getattr(self, feature_key, None)
          w = weights[feature_key]
          if val is True:
              vec[j] = w
          elif val is False:
              vec[j] = -w
          else:
              vec[j] = 0.0
      
      return vec

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

def ipa_to_segments(ipa: str) -> list[Phoneme]:
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

        # grab base char + following combining marks
        j = i + 1
        while j < len(ipa) and unicodedata.combining(ipa[j]):
            j += 1

        symbol = ipa[i:j]

        # palatalized consonants
        if j < len(ipa) and ipa[j] == PALATAL_MARKER:
            symbol += PALATAL_MARKER
            j += 1

        # append phoneme
        segments.append(get_phoneme(symbol, stress=stress))
        stress = False
        i = j

    return segments

@lru_cache(maxsize=100000)
def cached_text_to_segments(segment: str, lang: str):
    return ipa_to_segments(text_to_ipa(segment, lang))

WORD_RE = re.compile(r"\S+")

def find_phoneme_to_text_mapping(
        plaintext: str,
        phonemized: list[Phoneme],
        lang: str = "pl"
    ) -> list[dict["phoneme_index" | "plaintext_start" | "plaintext_end", int | str]]:
    """
    Find beginning and ending indices of text characters in plaintext, corresponding to Phonemes in list

    It is done using greedy word-level alignment

    Returned values have format:
    [{ "phoneme_index": int, "plaintext_index_start": int, "plaintext_index_end": int }, ...]
    """
    # extract words with spans
    words = [
        (m.group(), m.start(), m.end())
        for m in WORD_RE.finditer(plaintext)
    ]
    result = []
    phon_index = 0

    for word, w_start, w_end in words:
        word_ipa = text_to_ipa(word, lang=lang)
        print(word_ipa)
        word_segments = ipa_to_segments(word_ipa)

        aligned = align_word_to_phonemes(
            plaintext_word=word,
            phonemized_word=word_segments,
            lang=lang
        )

        if not aligned:
            # as fallback use whole word for all phonemes
            for idx in range(phon_index, phon_index + len(word_segments)):
                result.append({
                    "phoneme_index": idx,
                    "plaintext_start": w_start,
                    "plaintext_end": w_end,
                })
            phon_index += len(word_segments)
            continue

        # aligned pieces are in word-local indices
        for local_idx, span in enumerate(aligned):
            global_idx = phon_index + local_idx

            result.append({
                "phoneme_index": global_idx,
                "plaintext_start": w_start + span["word_index_start"],
                "plaintext_end": w_start + span["word_index_end"],
            })

        phon_index += len(word_segments)

    return result

def align_word_to_phonemes(
    plaintext_word: str,
    phonemized_word: list[Phoneme],
    lang: str = "pl",
    beam_width: int = 5,
    max_expand: int = 6
) -> list[dict["word_index_start" | "word_index_start", int]]:
    """
    List in same order as Phonemes in phonemized_word, containing start and end of phonemes
    [{ "word_index_start": int, "word_index_start": int }, ...]
    """

    def _phoneme_match_cost(a: list[Phoneme], b: list[Phoneme]) -> float:
        """
        Simple mismatch score, with symbol exact match
        0 is perfect match, higher cost is worse
        """
        n = max(len(a), len(b))
        cost = 0.0

        for i in range(n):
            if i >= len(a) or i >= len(b):
                cost += 1.0
                continue
            if a[i].symbol != b[i].symbol:
                cost += 1.0

        return cost

    n_chars = len(plaintext_word)
    n_phon = len(phonemized_word)

    initial = { "char_i": 0, "phon_i": 0, "path": [] }
    beam = [initial]

    finished = []

    while beam:
        new_beam = []

        for state in beam:

            if state["char_i"] == n_chars and state["phon_i"] == n_phon:
                finished.append(state)
                continue

            if state["char_i"] >= n_chars or state["phon_i"] >= n_phon:
                continue

            # expand character span
            for j in range(state["char_i"] + 1, min(n_chars + 1, state["char_i"] + max_expand + 1)):

                segment = plaintext_word[state["char_i"]:j]

                ipa = cached_text_to_segments(segment, lang=lang)

                # try aligning to next phoneme chunk sizes
                for k in range(1, min(n_phon - state["phon_i"] + 1, 6)):

                    target = phonemized_word[state["phon_i"]:state["phon_i"] + k]

                    cost = _phoneme_match_cost(ipa, target)

                    new_path = state["path"] + [(state["char_i"], j, state["phon_i"], state["phon_i"] + k, cost)]

                    new_beam.append(
                        { "char_i": j, "phon_i": state["phon_i"] + k, "path": new_path }
                    )

        # prune beam
        beam = sorted(new_beam, key=lambda s: len(s["path"]))[:beam_width]

    if not finished:
        return []

    best = min(finished, key=lambda s: len(s["path"]))

    # reconstruct output
    result = []
    for (ci, cj, pi, pj, cost) in best["path"]:
        result.append({
            "word_index_start": ci,
            "word_index_end": cj,
        })

    return result

@dataclass(frozen=True)
class MotifSpan:
    start: int
    end: int # exclusive

    match_start: int
    match_end: int # exclusive

    distance: float
    length: int

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
    Returns motif candidates across multiple window lengths
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
                    length=m,
                )
            )

    return candidates

def greedy_keep_longest_non_subset(candidates: list[MotifSpan]) -> list[MotifSpan]:
    """
    Remove candidates which are subsets of longer spans
    This also takes distance scores of pairs into account
    If adding more phonemes does not cause match to be worse then it is kept
    """
    candidates = sorted(
        candidates,
        key=lambda c: (round(c.distance, 5), -c.length, c.start, c.match_start) # Without rounding there is floating point noise in distance scores
    )

    kept: List[MotifSpan] = []

    def is_subset_of_kept(c: MotifSpan) -> bool:
        for k in kept:
            if k.covers(c):
                return True
            if k.covers(MotifSpan(c.match_start, c.match_end, c.start, c.end, c.distance, c.length)):
                return True
        return False

    for c in candidates:
        if not is_subset_of_kept(c):
            kept.append(c)

    return kept

if __name__ == "__main__":
    sample = "chrząszcz brzmi w trzcinie"
    ipa_phonemes = ipa_to_segments(text_to_ipa(sample))
    print(ipa_phonemes)

    # index_mapping = find_phoneme_to_text_mapping(sample, ipa_phonemes)
    # print(index_mapping)

    text_phoneme_mapping = find_phoneme_to_text_mapping(sample, ipa_phonemes)

    for phoneme_source, phoneme in zip(text_phoneme_mapping, ipa_phonemes):
        start = phoneme_source["plaintext_start"]
        end = phoneme_source["plaintext_end"]
        print(f"{phoneme.symbol} -> {sample[start:end]!r}")

    # vectors_representation = np.array([phoneme.as_vector() for phoneme in ipa_phonemes], dtype=np.float64)
    # print(vectors_representation)
    # vectors_for_stumpy = vectors_representation.T
    # X = vectors_representation.T
    # cands = extract_pairwise_motifs(X, m_values=range(3, 9), top_k_per_window=30)
    # kept = greedy_keep_longest_non_subset(cands)
    # for c in kept:
    #     print(c.distance, c.length, ipa_phonemes[c.start:c.end], ipa_phonemes[c.match_start:c.match_end])

# chrząszcz brzmi w trzcinie
# xʃˈɔ̃ʒdʒ bʒmˈi f tʃtɕˈiɲʲɛ
# [x, ʃ, ˈɔ̃, ʒ, d, ʒ, b, ʒ, m, ˈi, f, t͡ʃ, t͡ɕ, ˈi, ɲʲ, ɛ]

# TODO
# Original text of pairs would be found and motifs visualized
# Improve using a measurement rather than fixed amount of top pairs

# Other more expensive algorithms could be ran on each window to reveal which have least distance in respect to some distance measure
# Support patterns which have different lengths because of some insertions or deletions - Needleman–Wunsch or Smith–Waterman but not with equality but with feature vectors, Dynamic Time Warping, Soft-DTW
# Support patterns which have a few swaps - Damerau-style alignment, elastic matching
# On these generated pairs, some unified scoring would be used for final score of pairs, it could be expensive because there should not be that many of them
