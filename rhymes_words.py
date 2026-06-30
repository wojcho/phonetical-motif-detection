from phonemizer import phonemize
import panphon.distance
import regex
import spacy
from itertools import combinations

def pl_to_ipa(text: str) -> str:
    ipa = phonemize(
        text,
        language="pl",
        backend="espeak",
        strip=True
    )
    return ipa

dst = panphon.distance.Distance()

def distance_between_ipa_texts(a: str, b: str) -> str:
    distance_score = dst.weighted_feature_edit_distance(a, b)
    return distance_score

nlp = spacy.load("pl_core_news_sm")

def tokenize_words(text: str) -> list[str]:
    doc = nlp(text)
    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_space and not token.is_punct
    ]
    return tokens

def find_rhymes(text: str) -> dict:
    words = tokenize_words(text)
    # TODO this is done for now for words only, later for word pairs and longer sequences
    words_ipa = [pl_to_ipa(word) for word in words]
    words_ipa_deduplicated = set(words_ipa)
    results = []
    for word_a, word_b in combinations(words_ipa_deduplicated, 2):
        score = dst.weighted_feature_edit_distance(word_a, word_b)
        normalizer_word_length = 0.5 * (max(1, len(word_a)) + max(1, len(word_b))) # average length of word in pair
        normalized_score = score / normalizer_word_length
        results.append({
            "word_a": word_a,
            "word_b": word_b,
            "distance": normalized_score,
        })
    results_sorted = sorted(results, key=lambda x: x["distance"])
    return results_sorted

text = "Zbigniew Stonoga zjada pieroga. Niszczyć CORS. Crazy Wolf."
print(find_rhymes(text))
