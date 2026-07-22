from dataclasses import dataclass
import numpy as np
import stumpy

from phonemize import plaintext_to_phoneme_occurences

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

    def is_covering(self, other: MotifSpan) -> bool:
        return self.start <= other.start and self.end >= other.end and self.match_start <= other.match_start and self.match_end >= other.match_end

def extract_pairwise_motifs(X: np.ndarray, m_values: Iterable=range(2, 9), top_k_per_window: int=20) -> list[MotifSpan]:
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
            if k.is_covering(c):
                return True
            if k.is_covering(MotifSpan(c.match_start, c.match_end, c.start, c.end, c.distance, c.length)):
                return True
        return False

    for c in candidates:
        if not is_subset_of_kept(c):
            kept.append(c)

    return kept

if __name__ == "__main__":
    text = "W Szczebrzeszynie chrząszcz brzmi w trzcinie i Szczebrzeszyn z tego słynie"
    phonemized = plaintext_to_phoneme_occurences(text, "pol")

    for phoneme_occurence in phonemized:
        begin = phoneme_occurence.plaintext_index_begin
        end = phoneme_occurence.plaintext_index_end
        if phoneme_occurence.phoneme:
            print(f"{phoneme_occurence.phoneme.symbol} <- {text[begin:end]!r}")
        else:
            print(f"∅ <- {text[begin:end]!r}")

    # vectors_representation = np.array([phoneme.as_vector() for phoneme in ipa_phonemes], dtype=np.float64)
    # print(vectors_representation)
    # vectors_for_stumpy = vectors_representation.T
    # X = vectors_representation.T
    # cands = extract_pairwise_motifs(X, m_values=range(3, 9), top_k_per_window=30)
    # kept = greedy_keep_longest_non_subset(cands)
    # for c in kept:
    #     print(c.distance, c.length, ipa_phonemes[c.start:c.end], ipa_phonemes[c.match_start:c.match_end])

# TODO
# Showcase locations of found spans
# Improve using a measurement rather than fixed amount of top pairs

# Other more expensive algorithms could be ran on each window to reveal which have least distance in respect to some distance measure
# Support patterns which have different lengths because of some insertions or deletions - Needleman–Wunsch or Smith–Waterman but not with equality but with feature vectors, Dynamic Time Warping, Soft-DTW
# Support patterns which have a few swaps - Damerau-style alignment, elastic matching
# On these generated pairs, some unified scoring would be used for final score of pairs, it could be expensive because there should not be that many of them
