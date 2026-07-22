from g2p.mappings import Mapping, Rule
from g2p.transducer import Transducer
import json
from phoneme import Phoneme, PhonemeOccurence, IPA_FEATURES, IPA_TRIE

def prepare_transducer(language_iso_639_3_code: str = "pol") -> Transducer:
    if language_iso_639_3_code == "pol":
        with open("pol_to_ipa.json", "r") as file:
            polish_mapping = json.load(file)
    transducer = Transducer(Mapping(
        rules=polish_mapping,
        case_sensitive=False,
        preserve_case=False,
        norm_form="NFC",
        rule_ordering="as-written"
    ))
    return transducer

def tokenize_ipa(text: str):
    i = 0

    while i < len(text):
        node = IPA_TRIE
        j = i

        last_symbol = None
        last_end = i

        while j < len(text):
            ch = text[j]
            node = node.children.get(ch)
            if node is None:
                break

            j += 1

            if node.symbol is not None:
                last_symbol = node.symbol
                last_end = j

        if last_symbol is None:
            yield (i, text[i]) # unknown character
            i += 1
        else:
            yield (i, last_symbol)
            i = last_end

def plaintext_to_phoneme_occurences(
    plaintext: str,
    language_iso_639_3_code: str = "pol",
) -> list[PhonemeOccurence]:
    transducer = prepare_transducer()
    transduced = transducer(plaintext)
    as_ipa = transduced.output_string

    # Output index -> list of plaintext indices which produced its corresponding text
    out_to_inputs: dict[int, list[int]] = dict()
    for in_i, out_i in transduced.edges:
        out_to_inputs.setdefault(out_i, []).append(in_i)
    
    # Output index -> tuple (min, max) of plaintext indices which produced its corresponding text
    out_to_inputs_begin_end: dict[int, tuple[int, int]] = dict()
    for output_index, input_indices in out_to_inputs.items():
        begin, end = min(input_indices), max(input_indices) + 1
        out_to_inputs_begin_end[output_index] = (begin, end)

    # Build PhonemeOccurence objects
    occurences: list[PhonemeOccurence] = []
    for out_begin, symbol in tokenize_ipa(as_ipa):
        out_end = out_begin + len(symbol)

        # Collect every plaintext index that contributed to this IPA symbol
        input_indices: list[int] = []
        for out_i in range(out_begin, out_end):
            input_indices.extend(out_to_inputs.get(out_i, []))

        if input_indices:
            plaintext_begin = min(input_indices)
            plaintext_end = max(input_indices) + 1
        else:
            # Should not happen
            plaintext_begin = 0
            plaintext_end = 0

        # Skip unrecognized symbols
        if symbol in IPA_FEATURES:
            phoneme = Phoneme.from_symbol(symbol)
            occurences.append(
                PhonemeOccurence(
                    phoneme=phoneme,
                    plaintext_index_begin=plaintext_begin,
                    plaintext_index_end=plaintext_end,
                )
            )

    return occurences
