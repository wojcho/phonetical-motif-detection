from phonemizer import phonemize
import re
import numpy as np
import stumpy
import unicodedata

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

def find_phoneme_to_text_mapping(
    plaintext: str,
    phonemized: list[Phoneme],
    lang: str = "pl"
) -> list[dict["phoneme_index" | "plaintext_start" | "plaintext_end", int | str]]:
    """
    Find beginning and ending indices of text characters in plaintext, corresponding to Phonemes in list
    It assumes that for each Phoneme, many or none graphemes from plaintext can be assigned
    Returned values have format:
    [{ "phoneme_index": int, "plaintext_index_start": int, "plaintext_index_end": int }, ...]
    """
    pass

@dataclass(frozen=True, slots=True)
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

    def covers(self, other: MotifSpan) -> bool:
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
    text = "W Szczebrzeszynie chrząszcz brzmi w trzcinie i Szczebrzeszyn z tego słynie"
    ipa_phonemes = ipa_to_segments(text_to_ipa(text))
    print(ipa_phonemes)

    text_phoneme_mapping = find_phoneme_to_text_mapping(text, ipa_phonemes)

    for phoneme_source, phoneme in zip(text_phoneme_mapping, ipa_phonemes):
        start = phoneme_source["plaintext_start"]
        end = phoneme_source["plaintext_end"]
        print(f"{phoneme.symbol} -> {text[start:end]!r}")

    # vectors_representation = np.array([phoneme.as_vector() for phoneme in ipa_phonemes], dtype=np.float64)
    # print(vectors_representation)
    # vectors_for_stumpy = vectors_representation.T
    # X = vectors_representation.T
    # cands = extract_pairwise_motifs(X, m_values=range(3, 9), top_k_per_window=30)
    # kept = greedy_keep_longest_non_subset(cands)
    # for c in kept:
    #     print(c.distance, c.length, ipa_phonemes[c.start:c.end], ipa_phonemes[c.match_start:c.match_end])

# TODO
# Refactor to use separate PhonemeOccurence from flyweight Phoneme
# Use G2P instead of phonemizer
# In PhonemeOccurence, store plaintext_index_begin and plaintext_index_end
# Showcase locations of found spans
# Improve using a measurement rather than fixed amount of top pairs

# Other more expensive algorithms could be ran on each window to reveal which have least distance in respect to some distance measure
# Support patterns which have different lengths because of some insertions or deletions - Needleman–Wunsch or Smith–Waterman but not with equality but with feature vectors, Dynamic Time Warping, Soft-DTW
# Support patterns which have a few swaps - Damerau-style alignment, elastic matching
# On these generated pairs, some unified scoring would be used for final score of pairs, it could be expensive because there should not be that many of them
