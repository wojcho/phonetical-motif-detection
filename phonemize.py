from g2p.mappings import Mapping, Rule
from g2p.transducer import Transducer
import json
from phoneme import Phoneme, PhonemeOccurence, IPA_FEATURES

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

def plaintext_to_phoneme_occurences(
    plaintext: str,
    language_iso_639_3_code: str = "pol",
) -> list[PhonemeOccurence]:
    transducer = prepare_transducer()
    transduced = transducer(plaintext)
    as_ipa = transduced.output_string

    # output index -> list of plaintext indices which produced its corresponding text
    out_to_inputs: dict[int, list[int]] = dict()
    for in_i, out_i in transduced.edges:
        out_to_inputs.setdefault(out_i, []).append(in_i)
    
    # output index -> tuple (min, max) of plaintext indices which produced its corresponding text
    out_to_inputs_begin_end: dict[int, tuple[int, int]] = dict()
    for output_index, input_indices in out_to_inputs.items():
        begin, end = min(input_indices), max(input_indices) + 1
        out_to_inputs_begin_end[output_index] = (begin, end)

    # build PhonemeOccurence objects
    occurences: list[PhonemeOccurence] = []
    for out_i, ch in enumerate(as_ipa):
        begin, end = out_to_inputs_begin_end[out_i]
        if ch in IPA_FEATURES:
            phoneme = Phoneme.from_symbol(ch)
            occurences.append(
                PhonemeOccurence(
                    phoneme=phoneme,
                    plaintext_index_begin=begin,
                    plaintext_index_end=end,
                )
            )
        else:
            occurences.append(
                PhonemeOccurence(
                    phoneme=None,
                    plaintext_index_begin=begin,
                    plaintext_index_end=end,
                )
            )
    return occurences
