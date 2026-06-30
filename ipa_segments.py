from phonemizer import phonemize

def text_to_ipa(text: str, lang: str = "pl") -> str:
    """
    Convert text to IPA using espeak backend via phonemizer.
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


from dataclasses import dataclass
from typing import Dict, Optional

import re

@dataclass
class Phoneme:
    symbol: str
    stress: bool = False
    features: Optional[Dict[str, float]] = None

AFFRICATES = ["t͡s", "d͡z", "t͡ʂ", "d͡ʐ"]
PALATAL_MARKER = "ʲ"

AFFRICATE_NORMALIZATION = [
    (r"t\s?͡?\s?s", "t͡s"),
    (r"d\s?͡?\s?z", "d͡z"),
    (r"t\s?͡?\s?ʂ", "t͡ʂ"),
    (r"d\s?͡?\s?ʐ", "d͡ʐ"),
]

def normalize_ipa(ipa: str) -> str:
    for pattern, repl in AFFRICATE_NORMALIZATION:
        ipa = re.sub(pattern, repl, ipa)
    return ipa

def ipa_to_segments(ipa: str):
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
                segments.append((aff, stress))
                stress = False
                i += len(aff)
                matched = True
                break
        if matched:
            continue

        # palatalized consonants (simple heuristic)
        if i + 1 < len(ipa) and ipa[i+1] == PALATAL_MARKER:
            segments.append((ipa[i] + "ʲ", stress))
            stress = False
            i += 2
            continue

        # normal phoneme
        segments.append((ipa[i], stress))
        stress = False
        i += 1

    return segments

if __name__ == "__main__":
    sample = "Znowu staniesz w kolejce"
    print(ipa_to_segments(text_to_ipa(sample)))
