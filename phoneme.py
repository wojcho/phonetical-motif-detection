from dataclasses import dataclass, field
import csv
import numpy as np

def load_feature_weights(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items() if v not in ("", None)}

FEATURE_WEIGHTS = load_feature_weights("./feature_weights.csv")

_PHONEME_CACHE: dict[str, Phoneme] = {}

@dataclass(frozen=True, slots=True)
class Phoneme:
    symbol: str

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
    def from_symbol(symbol: str) -> Phoneme:
        """
        For flyweight pattern, to not duplicate Phonemes in memory
        """
        p = _PHONEME_CACHE.get(symbol)
        if p is None:
            p = Phoneme._from_symbol(symbol)
            _PHONEME_CACHE[symbol] = p
        return p

    @staticmethod
    def _from_symbol(symbol: str) -> Phoneme:
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
            if attr_name in ("symbol"):
                continue
            if attr_name in row:
                kwargs[attr_name] = parse_cell(row[attr_name])

        return Phoneme(symbol=symbol, **kwargs)

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
        return self.symbol

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

@dataclass(frozen=True, slots=True)
class PhonemeOccurence:
    phoneme: Phoneme

    plaintext_index_begin: int # Index of first index which was transduced from plaintext part to phoneme
    plaintext_index_end: int # Index after last index which was transduced from plaintext part to phoneme

IPA_SYMBOLS = sorted(
    IPA_FEATURES.keys(),
    key=len,
    reverse=True,
)

@dataclass(slots=True)
class TrieNode:
    children: dict[str, TrieNode] = field(default_factory=dict)
    symbol: str | None = None

def build_trie(symbols: list[str]) -> TrieNode:
    root = TrieNode()

    for symbol in symbols:
        node = root
        for ch in symbol:
            node = node.children.setdefault(ch, TrieNode())
        node.symbol = symbol

    return root

IPA_TRIE = build_trie(IPA_SYMBOLS)
