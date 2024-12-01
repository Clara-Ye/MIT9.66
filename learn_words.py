import json
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm

def form_association(word, word_stem, freq, assoc):
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

    # add a new entry if the word is already associated with the word stem
    if not any(entry["word"] == word for entry in curr_assoc):
        curr_assoc.append({"word": word, "length": len(word), "freq": freq})
    # update the frequency if the word is already associated
    else:
        for entry in curr_assoc:
            if entry["word"] == word:
                entry["freq"] += freq
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
                form_association(word, word_stem, freq, assoc)

    # special associations for the start and end of the word
    if (np.random.rand() < p_assoc):
        form_association(word, f"{word[0]}*", freq, assoc) # start marker
    if (np.random.rand() < p_assoc):
        form_association(word, f"*{word[-1]}", freq, assoc) # end marker


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
    learn_words_from_corpus(eta=0.6, output_path="data/word_associations_06.json")
