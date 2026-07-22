from dataclasses import dataclass
import numpy as np
import stumpy

from phonemize import plaintext_to_phoneme_occurences

@dataclass(frozen=True, slots=True)
class MotifSpan:
    begin: int
    end: int # exclusive

    match_begin: int
    match_end: int # exclusive

    distance: float
    length: int

    @property
    def interval(self) -> tuple[int, int]:
        return (self.begin, self.end)

    @property
    def match_interval(self) -> tuple[int, int]:
        return (self.match_begin, self.match_end)

    def is_covering(self, other: MotifSpan) -> bool:
        return self.begin <= other.begin and self.end >= other.end and self.match_begin <= other.match_begin and self.match_end >= other.match_end
    
    def to_plaintext_span(self, phoneme_occurences: list[PhonemeOccurence]) -> tuple[tuple[int, int], tuple[int, int]]:
        relevant = phoneme_occurences[self.begin:self.end]
        match_relevant = phoneme_occurences[self.match_begin:self.match_end]

        span = (
            min(p.plaintext_index_begin for p in relevant),
            max(p.plaintext_index_end for p in relevant),
        )

        match_span = (
            min(p.plaintext_index_begin for p in match_relevant),
            max(p.plaintext_index_end for p in match_relevant),
        )

        return span, match_span
        # # This assumes that phoneme_occurences are guaranteed to be ordered by plaintext position and never overlap
        # return (
        #     (
        #         phoneme_occurences[self.begin].plaintext_index_begin,
        #         phoneme_occurences[self.end - 1].plaintext_index_end,
        #     ),
        #     (
        #         phoneme_occurences[self.match_begin].plaintext_index_begin,
        #         phoneme_occurences[self.match_end - 1].plaintext_index_end,
        #     ),
        # )

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
                    begin=int(i),
                    end=int(i + m),
                    match_begin=j,
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
        key=lambda c: (round(c.distance, 5), -c.length, c.begin, c.match_begin) # Without rounding there is floating point noise in distance scores
    )

    kept: List[MotifSpan] = []

    def is_subset_of_kept(c: MotifSpan) -> bool:
        for k in kept:
            if k.is_covering(c):
                return True
            if k.is_covering(MotifSpan(c.match_begin, c.match_end, c.begin, c.end, c.distance, c.length)):
                return True
        return False

    for c in candidates:
        if not is_subset_of_kept(c):
            kept.append(c)

    return kept

@dataclass(frozen=True, slots=True)
class PlaintextSpan:
    plaintext_index_begin: int
    plaintext_index_end: int # exclusive

    plaintext_index_match_begin: int
    plaintext_index_match_end: int # exclusive

    match_distance_score: float

    plaintext: str # Python strings are pointers to cache, so this should not increase memory usage

    @property
    def text(self) -> str:
        return self.plaintext[self.plaintext_index_begin:self.plaintext_index_end]

    @property
    def match_text(self) -> str:
        return self.plaintext[
            self.plaintext_index_match_begin:self.plaintext_index_match_end
        ]

    def __str__(self) -> str:
        return f"({self.match_distance_score:.4f} {self.text!r} <=> {self.match_text!r})"
    
    def __repr__(self) -> str:
        return self.__str__()

def plaintext_to_motif_plaintext_spans(plaintext: str, language_iso_639_3_code: str = "pol") -> list[PlaintextSpan]:
    # Convert text to list[PhonemeOccurence] which retains information about correspondence of plaintext to IPA
    phonemized = plaintext_to_phoneme_occurences(plaintext, "pol")

    # for phoneme_occurence in phonemized:
    #     begin = phoneme_occurence.plaintext_index_begin
    #     end = phoneme_occurence.plaintext_index_end
    #     if phoneme_occurence.phoneme:
    #         print(f"{phoneme_occurence.phoneme.symbol} <- {plaintext[begin:end]!r}")
    #     else:
    #         print(f"∅ <- {plaintext[begin:end]!r}")

    # Find motifs
    vectors_representation = np.array([phoneme_occurence.phoneme.as_vector() for phoneme_occurence in phonemized], dtype=np.float64)
    # print(vectors_representation)
    vectors_for_stumpy = vectors_representation.T
    X = vectors_representation.T
    cands = extract_pairwise_motifs(X, m_values=range(3, 9), top_k_per_window=30)
    kept = greedy_keep_longest_non_subset(cands)

    # Process outputs back to use plaintext indices
    plaintext_spans = []
    for c in kept:
        plaintext_span = c.to_plaintext_span(phonemized)
        plaintext_spans.append(PlaintextSpan(
            plaintext_index_begin=plaintext_span[0][0],
            plaintext_index_end=plaintext_span[0][1],
            plaintext_index_match_begin=plaintext_span[1][0],
            plaintext_index_match_end=plaintext_span[1][1],
            match_distance_score=c.distance,
            plaintext=plaintext,
        ))
    return plaintext_spans

if __name__ == "__main__":
    print(plaintext_to_motif_plaintext_spans("W Szczebrzeszynie chrząszcz brzmi w trzcinie i Szczebrzeszyn z tego słynie", "pol"))

# TODO
# Improve using a measurement rather than fixed amount of top pairs
