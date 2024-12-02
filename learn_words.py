import json
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm

def form_association(word, word_stem, freq, assoc, position, nu=0.9):
    """
    Inserts or updates an association between a word and a word stem.

    Args:
        word (str): The complete word.
        word_stem (str): The word stem (e.g., a bigram or a special marker).
        freq (int): Frequency of the word in the corpus.
        assoc (dict): The dictionary storing associations.      
    """
    # get current associations for the word stem, or initialize an empty list
    curr_assoc = assoc[word_stem] if word_stem in assoc else []

    # compute attention weight based on word frequency and word stem serial position:
    attention_weight = freq * (nu ** (position - 1))

    # add a new entry if the word is already associated with the word stem
    if not any(entry["word"] == word for entry in curr_assoc):
        curr_assoc.append({"word": word, "length": len(word), "freq": attention_weight})
    # update the frequency if the word is already associated
    else:
        for entry in curr_assoc:
            if entry["word"] == word:
                entry["freq"] += attention_weight
                break

    # update the associations dictionary in-place
    assoc[word_stem] = curr_assoc


def learn_associations(word, freq, assoc, eta=0.0):
    """
    Learns associations between a word and its stems (e.g., bigrams).

    Args:
        word (str): The word to analyze.
        freq (int): Frequency of the word in the corpus.
        assoc (dict): The dictionary to store associations.
    """
    # calculate the probability of forming an association based on word length
    wl = len(word)
    p_assoc = (1.0 - eta * np.log(wl - 3)) if wl > 3 else 1.0
    p_assoc = np.clip(p_assoc, 0.0, 1.0) # ensure valid probability

    # bigrams or shorter word stems (max length: 2)
    for i in range(wl):
        for j in range(i+1, min(wl+1, i+3)):
            word_stem = word[i:j]
            # form association with probability
            if (np.random.rand() < p_assoc):
                form_association(word, word_stem, freq, assoc, position=i+1)

    # special associations for the start and end of the word
    if (np.random.rand() < p_assoc):
        form_association(word, f"{word[0]}*", freq, assoc, position=1) # start marker
    if (np.random.rand() < p_assoc):
        form_association(word, f"*{word[-1]}", freq, assoc, position=len(word)) # end marker


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
    corpus = pd.read_csv(corpus_path, na_values=[], keep_default_na=False)

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
    eden = learn_associations("EDEN", 3, dict())
    clara = learn_associations("CLARA", 1, eden)
    claire = learn_associations("CLAIRE", 2, clara)
    pprint.pprint(claire)
    with open("test.json", "w") as file:
        json.dump(claire, file, indent=4)

if __name__ == "__main__":
    #test()
    learn_words_from_corpus(eta=0.0, output_path="data/word_associations_00.json")
    learn_words_from_corpus(eta=0.2, output_path="data/word_associations_02.json")
    learn_words_from_corpus(eta=0.4, output_path="data/word_associations_04.json")
    learn_words_from_corpus(eta=0.6, output_path="data/word_associations_06.json")
