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


def retrieve_next_valid(word_stems, words, probs, searched_words, prob_threshold=0.0):
    """
    Retrieves the next valid word using parallel lists for words and probabilities.

    Args:
        word_stems (list): List of word stems that must be present in the word.
        words (list): List of candidate words.
        probs (list): List of probabilities corresponding to the candidate words.
        searched_words (set): Set of words already searched and validated.

    Returns:
        str: The next valid word that matches the criteria, or None if no valid answer is found.
    """
    for word, prob in zip(words, probs):
        # Skip already searched words or those below the probability threshold
        if word in searched_words or prob < prob_threshold:
            continue

        searched_words.add(word)  # mark this word as searched

        # Check if the word matches all required word stems
        is_valid = True
        for word_stem in word_stems:
            if word_stem.endswith("*"):  # handle starting letter match (e.g., "X*")
                if not word.startswith(word_stem[:-1]):
                    is_valid = False
                    break
            elif word_stem.startswith("*"):  # handle ending letter match (e.g., "*X")
                if not word.endswith(word_stem[1:]):
                    is_valid = False
                    break
            elif word_stem not in word:  # general case
                is_valid = False
                break

        if is_valid:
            return word, searched_words

    print("No valid answer retrieved.")
    return None, None


def find_answer(word_stems, candidate_probs, ground_truth, prob_threshold=0.0, max_search=float('inf')):
    """
    Loops until the correct answer is found using the retrieve_next_valid_parallel function.

    Args:
        word_stems (list): List of word stems that must be present in the word.
        candidate_probs (dict): Dictionary of candidate words with their probabilities.
        ground_truth (str): The correct answer to find.
        prob_threshold (float): Minimum probability threshold for a candidate to be considered.

    Returns:
        str: The correct answer, if found, or None if no valid answer could be retrieved.
    """
    # Split dictionary into separate lists for words and probabilities
    words, probs = zip(*candidate_probs.items())

    # Sort words and probabilities outside the loop
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    # Initialize set to track searched words
    searched_words = set()
    
    while len(searched_words) < max_search:
        next_word, searched_words = retrieve_next_valid(word_stems, sorted_words, sorted_probs, searched_words, prob_threshold)
        if not next_word:
            print("Failed to retrieve a valid answer.")
            return None

        # Check if the retrieved word matches the ground truth
        if next_word == ground_truth:
            print(f"Correct answer retrieved after {len(searched_words)} attempts: {next_word}")
            return next_word


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
    # Compute candidate scores
    candidate_probs = compute_candidate_scores(word_stems, target_length, associations)

    # Sort and select top candidates
    sorted_candidates = sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:top_n] if top_n else sorted_candidates

    return top_candidates


if __name__ == "__main__":
    with open("data/word_associations_06.json", "r") as file:
        associations = json.load(file)

    word_stems = ["C*", "OU"]
    target_length = 5
    candidate_probs = compute_candidate_scores(word_stems, target_length, associations)
    pprint.pprint(retrieve_top_candidates(word_stems, target_length, associations, top_n=20))

    find_answer(word_stems, candidate_probs, "CLOUD")
