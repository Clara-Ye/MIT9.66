import json
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_corpus(file_path):
    return pd.read_csv(file_path, na_values=[], keep_default_na=False)


def extract_word_stems(word):
    """
    Extracts word stems with rough positional tagging (e.g., 'FIRST_HALF', 'SECOND_HALF').

    Args:
        word (str): The input word.

    Returns:
        dict: Word stems mapped to rough positions.
    """
    wl = len(word)
    word = word.upper()
    word_stems = dict()

    # Split the word into two halves
    midpoint = wl // 2

    for i in range(wl):
        for j in range(i + 1, min(wl + 1, i + 3)):  # Extract stems of length 1 or 2
            word_stem = word[i:j]

            # Determine rough position
            if i < midpoint:
                position_tag = "FIRST_HALF"
            else:
                position_tag = "SECOND_HALF"

            # Use "|"-joined strings as keys
            word_stems[f"{word_stem}|{position_tag}"] = i

    return word_stems


def form_association(word, word_stem_key, freq, assoc, position, nu=0.9):
    """
    Inserts or updates an association between a word and a word stem with rough positions.

    Args:
        word (str): The complete word.
        word_stem (tuple): The word stem and its rough position (e.g., ('HE', 'FIRST_HALF')).
        freq (int): Frequency of the word in the corpus.
        assoc (dict): The dictionary storing associations.
    """
    curr_assoc = assoc.get(word_stem_key, [])

    # Compute attention weight based on word frequency and position
    attention_weight = freq * (nu ** (position - 1))

    # Check if word already associated with the stem
    if not any(entry["word"] == word for entry in curr_assoc):
        curr_assoc.append({"word": word, "length": len(word), "freq": attention_weight})
    else:
        for entry in curr_assoc:
            if entry["word"] == word:
                entry["freq"] += attention_weight
                break

    assoc[word_stem_key] = curr_assoc


def learn_associations(word, freq, assoc, eta=0.0):
    """
    Learns associations between a word and its stems with rough positions.

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
    Adds a 'prob' field to each association in the existing dictionary to represent normalized probabilities.

    Args:
        assoc (dict): The dictionary of associations with frequency counts.
    """
    for word_stem, entries in assoc.items():
        # Calculate the total frequency for the stem
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
    Learns associations from a word-frequency corpus and saves them to a file.

    Args:
        corpus_path (str): Path to the input CSV file containing the corpus.
        output_path (str): Path to save the resulting associations JSON file.
    """    
    assoc = dict()
    # load the corpus, treating "NULL" or other values as valid strings
    corpus = load_corpus(corpus_path)

    # learn associations for each word
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
    learn_words_from_corpus(eta=0.0, output_path="data/word_associations_with_pos_00.json")
    learn_words_from_corpus(eta=0.2, output_path="data/word_associations_with_pos_02.json")
    learn_words_from_corpus(eta=0.4, output_path="data/word_associations_with_pos_04.json")
    learn_words_from_corpus(eta=0.6, output_path="data/word_associations_with_pos_06.json")
