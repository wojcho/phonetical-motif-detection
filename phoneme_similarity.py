from phonemizer import phonemize
import re

def tokenize(text):
    tokens = re.findall(r"\w+", text, re.UNICODE)
    return [t for t in tokens if t.strip()]

def word_to_phonemes(word: str) -> list[str]:
    ipa = phonemize(
        word,
        language="pl",
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False
    )
    if not ipa:
        return []
    return list(ipa)

VOWEL_CLASS = {
    "a": "A", "ɑ": "A", "ɔ": "A",
    "e": "E", "ɛ": "E", "ɛ̃": "E",
    "i": "I", "ɨ": "I",
    "o": "O", "u": "O", "ɔ̃": "O",
}

CONSONANT_CLASS = {
    # nasals
    "m": "N", "n": "N", "ɲ": "N",

    # stops
    "p": "P", "b": "P",
    "t": "T", "d": "T",
    "k": "K", "g": "K",

    # fricatives / sibilants
    "s": "S", "z": "S",
    "ʃ": "S", "ʒ": "S",
    "x": "S","ɣ": "S",

    # labiodental fricatives
    "f": "F", "v": "F",

    # liquids
    "l": "L", "r": "L",

    # approximants
    "w": "W", "j": "J",
}

def ph_to_class(ph: str) -> str:
    if ph in VOWEL_CLASS:
        return "V:" + VOWEL_CLASS[ph]
    if ph in CONSONANT_CLASS:
        return "C:" + CONSONANT_CLASS[ph]
    return None

def word_to_classes(word):
    phs = word_to_phonemes(word)
    out = []
    for p in phs:
        c = ph_to_class(p)
        if c is not None:
            out.append(c)
    # print(word, phs, out)
    return out

def build_spans(words, max_len=3):
    spans = []

    for i in range(len(words)):
        for L in range(1, max_len + 1):
            if i + L <= len(words):
                span_words = words[i:i+L]

                span_classes = []
                for w in span_words:
                    span_classes.extend(word_to_classes(w))

                spans.append({
                    "text": " ".join(span_words),
                    "classes": span_classes,
                    "start": i,
                    "end": i + L - 1
                })

    return spans

def class_sim(a, b):
    # identical class
    if a == b:
        # vowels count stronger
        if a.startswith("V:") and b.startswith("V:"):
            return 2.0
        return 1.0

    # handle unrecognized
    if a is None or b is None:
        return -0.5

    # vowel-vowel mismatch (still related)
    if a.startswith("V:") and b.startswith("V:"):
        return 0.2

    # consonant same manner group
    if a.startswith("C:") and b.startswith("C:"):
        if a[2] == b[2]:
            return 0.4
        return 0.1

    # vowel vs consonant
    return -0.2

def alignment_score(seq1, seq2):
    n, m = len(seq1), len(seq2)

    # Dynamic Programming matrix, Levenshtein-style
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    gap_penalty = -0.4

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1][j-1] + class_sim(seq1[i-1], seq2[j-1])
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty

            dp[i][j] = max(match, delete, insert)

    if n == 0 and m == 0:
        return 0

    return dp[n][m] / (n + m)

def overlap(a, b):
    return not (a["end"] < b["start"] or b["end"] < a["start"])

def find_matches(spans, window=10, threshold=0.3):
    matches = []

    for i in range(len(spans)):
        for j in range(i+1, min(i+window, len(spans))):
            s1 = spans[i]
            s2 = spans[j]

            # Skip diagonals
            if overlap(s1, s2):
                continue

            score = alignment_score(s1["classes"], s2["classes"])

            if score >= threshold:
                matches.append((s1["text"], s2["text"], score))

    return matches

def analyze_text(text):
    words = tokenize(text)
    print(words)
    spans = build_spans(words, max_len=3)
    print(spans)
    return find_matches(spans)

# Example usage

text = """
Nie chcę spadać w otchłań, ale biegnę z górki
Mam się czołgać, jest stroma, nie chcę wrócić
I nie mogę się wycofać, kiedy mogę
Kiedy mogę, kiedy mogę

Emocje biegną po dachach, mieszkamy niżej
Nie mogę często odmawiać, bo stracę wszystko
Chcę biec razem z nimi, oddziaływać
Nie mieć żadnej siły, aż donikąd
"""

# What it is supposed to detect

# otchłań - czołgać - stroma
# biegnę - nie chcę
# górki - wrócić

# emocje - nie mogę
# biegną - często
# po dachach - odmawiać
# chcę biec - nie mieć
# żadnej siły - razem z nimi
# wszystko - donikąd

matches = analyze_text(text)

print(matches)
for a, b, score in sorted(matches, key=lambda x: x[-1], reverse=True):
    print(f"{a:20} <-> {b:20}  {score:.2f}")

# TODO deduplicate, leave only those with longest strings and not matches between their substrings
# TODO check parts of speech
# TODO penalize exact matches
# TODO provide annotations for text
