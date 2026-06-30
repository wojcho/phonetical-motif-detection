from phonemizer import phonemize
import re
import csv
from dataclasses import dataclass

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

        # Map CSV columns to dataclass fields
        field_map = {
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

        kwargs: dict[str, bool | None] = {}
        for csv_col, attr_name in field_map.items():
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

if __name__ == "__main__":
    sample = "więcej wybór burza ciąg ćwiek śnieg wątek źródło ziarno żywot"
    print(ipa_to_segments(text_to_ipa(sample)))
