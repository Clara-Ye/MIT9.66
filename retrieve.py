import json
import numpy as np
import pprint


def adjust_probabilities_by_length(candidate_probs, target_length):
    """
    Adjusts candidate probabilities based on word length proximity to the target length.

    Args:
        candidate_probs (dict): Dictionary of candidate words with their probabilities.
        target_length (int): Target word length to favor.
    """
    target_length_log = np.log(target_length)
    for word, prob in candidate_probs.items():
        word_length_log = np.log(len(word))
        length_diff = abs(word_length_log - target_length_log)
        if length_diff == 0:
            candidate_probs[word] *= 3  # Favor exact match
        elif length_diff > 1:
            candidate_probs[word] = 0  # Penalize large deviations


def normalize(candidate_probs):
    """
    Normalizes candidate probabilities to sum to 1.

    Args:
        candidate_probs (dict): Dictionary of candidate words with their raw probabilities.

    Returns:
        dict: Normalized probabilities.
    """
    total_prob = sum(candidate_probs.values())

    if total_prob > 0:
        for word in candidate_probs:
            candidate_probs[word] /= total_prob
    else:
        for word in candidate_probs:
            candidate_probs[word] = 0


def compute_candidate_scores(word_stems, target_length, associations, sigma=1e-5):
    """
    Computes candidate scores based on orthographic stems and probabilities.

    Args:
        word_stems (list): List of word stems to consider.
        target_length (int): Target word length.
        associations (dict): Precomputed word-stem associations.
        smoothing (float): Smoothing constant to avoid zero probabilities.

    Returns:
        dict: Candidate words and their scores.
    """
    candidate_probs = {}

    # Aggregate probabilities across all stems
    for stem in word_stems:
        if (stem not in associations):
            print(f"Stem '{stem}' has no associations.")
            continue
        for entry in associations[stem]:
            word = entry["word"]
            candidate_probs[word] = candidate_probs.get(word, 1) * entry["prob"]

    # Apply smoothing and normalize by number of stems
    n_stems = len(word_stems)
    candidate_probs = {
        word: (prob + sigma) ** (1 / n_stems) for word, prob in candidate_probs.items()
        }

    # Adjust by length and normalize
    adjust_probabilities_by_length(candidate_probs, target_length)
    normalize(candidate_probs)

    return candidate_probs


def retrieve_top_candidates(word_stems, target_length, associations, top_n=10):
    """
    Retrieves top candidate words based on the given stems and target length.

    Args:
        word_stems (list): List of word stems to consider.
        target_length (int): Target word length.
        associations (dict): Precomputed word-stem associations.
        top_n (int): Number of top candidates to return.

    Returns:
        list: List of top candidate words and their probabilities.
    """
    candidate_probs = compute_candidate_scores(word_stems, target_length, associations)
    return sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]


if __name__ == "__main__":
    with open("data/word_associations_02.json", "r") as file:
        associations = json.load(file)

    stems = ["C*", "OU"]
    target_length = 5
    top_candidates = retrieve_top_candidates(stems, target_length, associations, top_n=10)
    pprint.pprint(top_candidates)
