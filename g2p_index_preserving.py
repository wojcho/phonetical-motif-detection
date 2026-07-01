from g2p.mappings import Mapping, Rule
from g2p.transducer import Transducer
import json

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

text = "szaleństwo"
transducer = prepare_transducer()
transduced = transducer(text)
as_ipa = transduced.output_string

for out_i, out_ch in enumerate(as_ipa):
    inputs = [i for i, o in transduced.edges if o == out_i]
    if inputs:
        start, end = min(inputs), max(inputs) + 1
        in_str = text[start:end]
    else:
        in_str = "-"
    print(f"{out_ch} <- {in_str}")
