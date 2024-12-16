import json
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_corpus(file_path):
    """Loads the word corpus from a CSV file."""
    return pd.read_csv(file_path, na_values=[], keep_default_na=False)


def extract_word_stems(word):
    """
    Extracts word stems with rough positional tagging.

    Args:
        word (str): The input word.

    Returns:
        dict: Word stems mapped to rough positions.
    """
    wl = len(word)
    word = word.upper()
    word_stems = {}

    midpoint = wl // 2
    for i in range(wl):
        for j in range(i + 1, min(wl + 1, i + 3)):  # Extract stems of length 1 or 2
            word_stem = word[i:j]
            position_tag = "FIRST_HALF" if i < midpoint else "SECOND_HALF"
            word_stems[f"{word_stem}|{position_tag}"] = i

    return word_stems


def form_association(word, word_stem_key, freq, assoc, position, nu=0.9):
    """
    Inserts or updates an association between a word and its stem with rough positions.

    Args:
        word (str): The complete word.
        word_stem_key (str): The word stem key with position tag.
        freq (int): Frequency of the word in the corpus.
        assoc (dict): The dictionary storing associations.
        position (int): Position of the stem in the word.
        nu (float): Decay factor for positional weighting.
    """
    curr_assoc = assoc.get(word_stem_key, [])
    attention_weight = freq * (nu ** (position - 1))

    for entry in curr_assoc:
        if entry["word"] == word:
            entry["freq"] += attention_weight
            break
    else:
        curr_assoc.append({"word": word, "length": len(word), "freq": attention_weight})

    assoc[word_stem_key] = curr_assoc


def learn_associations(word, freq, assoc, eta=0.0):
    """
    Learns associations between a word and its stems.

    Args:
        word (str): The word to analyze.
        freq (int): Frequency of the word in the corpus.
        assoc (dict): The dictionary to store associations.
    """
    wl = len(word)
    p_assoc = (1.0 - eta * np.log(wl - 3)) if wl > 3 else 1.0
    p_assoc = np.clip(p_assoc, 0.0, 1.0) # ensure valid probability

    word_stems = extract_word_stems(word)

    for word_stem_key, position in word_stems.items():
        if np.random.rand() < p_assoc:
            form_association(word, word_stem_key, freq, assoc, position)

    # Add special start and end markers
    if np.random.rand() < p_assoc:
        form_association(word, f"{word[0]}*|FIRST_HALF", freq, assoc, position=1)
    if np.random.rand() < p_assoc:
        form_association(word, f"*{word[-1]}|SECOND_HALF", freq, assoc, position=wl)


def normalize_to_probabilities(assoc):
    """
    Normalizes frequencies to probabilities in the association dictionary.

    Args:
        assoc (dict): Dictionary of word-stem associations.
    """
    for _, entries in assoc.items():
        total_freq = sum(entry["freq"] for entry in entries)
        if total_freq > 0:
            # Add the normalized probability to each word's entry
            for entry in entries:
                entry["prob"] = entry["freq"] / total_freq
        else:
            # Assign 0 probability to all entries
            for entry in entries:
                entry["prob"] = 0.0


def learn_words_from_corpus(eta=0.0, corpus_path="data/thorndike_corpus.csv", output_path="data/word_associations.json"):
    """
    Learns word-stem associations from a corpus and saves them to a file.

    Args:
        eta (float): Decay factor for association formation.
        corpus_path (str): Path to the corpus CSV file.
        output_path (str): Path to save the learned associations as JSON.
    """
    assoc = {}
    corpus = load_corpus(corpus_path)

    # Learn associations for each word
    for i, row in tqdm(corpus.iterrows()):
        try: 
            learn_associations(row["Word"], row["Frequency"], assoc, eta=eta)
        except Exception as e:
            print(e)
            print(f"row {i}: {row}")
            continue

    # Normalize frequencies to probabilities
    normalize_to_probabilities(assoc)

    # print summary and save the associations to a file
    print(f"{sum(len(entry) for entry in assoc.values())} associations learned.")
    with open(output_path, "w") as file:
        json.dump(assoc, file, indent=4)
        print(f"File saved to {output_path}.")


def test():
    assoc = dict()
    learn_associations("EDEN", 3, assoc)
    learn_associations("CLARA", 1, assoc)
    learn_associations("CLAIRE", 2, assoc)
    pprint.pprint(assoc)
    with open("test.json", "w") as file:
        json.dump(assoc, file, indent=4)


if __name__ == "__main__":
    #test()
    learn_words_from_corpus(eta=0.0, output_path="data/word_associations_00.json")
    learn_words_from_corpus(eta=0.2, output_path="data/word_associations_02.json")
    learn_words_from_corpus(eta=0.4, output_path="data/word_associations_04.json")
    learn_words_from_corpus(eta=0.6, output_path="data/word_associations_06.json")
    learn_words_from_corpus(eta=0.8, output_path="data/word_associations_08.json")
