import json
import random
import string
import pprint
import numpy as np


def adjust_probabilities_by_length(candidate_probs, target_length):
    """
    Adjusts candidate probabilities based on word length proximity to the target length.

    Args:
        candidate_probs (dict): Dictionary of candidate words with their probabilities.
        target_length (int): Target word length to favor.
    """
    target_length_log = np.log(target_length)
    for word, _ in candidate_probs.items():
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
    """
    total_prob = sum(candidate_probs.values())
    if (total_prob > 0):
        for word in candidate_probs:
            candidate_probs[word] /= total_prob
    else:
        for word in candidate_probs:
            candidate_probs[word] = 0


def compute_gray_penalty(word, gray_letters, gray_penalty_factor):
    gray_count = sum(1 for char in word if char in gray_letters)
    gray_penalty = gray_penalty_factor ** gray_count
    return gray_penalty


def compute_candidate_scores(word_stem_keys, gray_letters, target_length, associations, sigma=1e-5, gray_penalty_factor=0.7, pos_penalty_factor=0.3):
    """
    Computes candidate scores based on orthographic stems and rough positional alignment.

    Args:
        word_stem_keys (list): List of word stems with rough positions (e.g., ['HE|FIRST_HALF']).
        target_length (int): Target word length.
        associations (dict): Precomputed word-stem associations.
        sigma (float): Smoothing constant to avoid zero probabilities.

    Returns:
        dict: Candidate words and their scores.
    """
    candidate_probs = {}

    if (word_stem_keys):
        # Aggregate probabilities across all stems
        for word_stem_key in word_stem_keys:
            if (word_stem_key not in associations):
                print(f"Stem '{word_stem_key}' has no associations.")
                continue
            for entry in associations[word_stem_key]:
                word = entry["word"]
                word_stem, rough_position = word_stem_key.split("|")
                
                # Calculate gray letter penalty
                gray_penalty = compute_gray_penalty(word, gray_letters, gray_penalty_factor)
                
                # Check rough positional alignment
                if (rough_position == "FIRST_HALF" and word.find(word_stem) <= len(word) // 2) or \
                (rough_position == "SECOND_HALF" and word.find(word_stem) > len(word) // 2):
                    candidate_probs[word] = candidate_probs.get(word, 1) * (entry["prob"] * pos_penalty_factor * gray_penalty)
                else:
                    candidate_probs[word] = candidate_probs.get(word, 1) * (entry["prob"] * gray_penalty)
    else:
        # If no word stems, consider all words in associations
        for word_stem_key, assoc in associations.items():
            for entry in assoc:
                word, prob = entry["word"], entry["prob"]
                if (len(word) == target_length):  # Ensure matching length
                    # Calculate gray letter penalty
                    gray_penalty = compute_gray_penalty(word, gray_letters, gray_penalty_factor)
                    candidate_probs[word] = prob * gray_penalty

    # Apply smoothing and normalize by number of stems
    n_stems = max(1, len(word_stem_keys))  # Avoid division by zero
    candidate_probs = {
        word: (prob + sigma) ** (1 / n_stems) for word, prob in candidate_probs.items()
    }

    # Adjust by length and normalize
    adjust_probabilities_by_length(candidate_probs, target_length)
    normalize(candidate_probs)

    return candidate_probs


def retrieve_next_valid(word_stem_keys, target_length, words, probs, searched_words, prob_threshold=0.0):
    """
    Retrieves the next valid word using rough positional alignment.

    Args:
        word_stem_keys (list): List of word stems with rough positions (e.g., ['HE|FIRST_HALF']).
        target_length (int): Target word length.
        words (list): List of candidate words.
        probs (list): List of probabilities corresponding to the candidate words.
        searched_words (set): Set of words already searched and validated.
        prob_threshold (float): Minimum probability threshold for valid candidates.

    Returns:
        tuple: The next valid word and updated searched words.
    """
    for word, prob in zip(words, probs):
        if word in searched_words or prob < prob_threshold:
            continue

        searched_words.add(word)

        # Check if the word matches all required word stems
        is_valid = True
        for word_stem_key in word_stem_keys:
            word_stem = word_stem_key.split("|")[0]
            # Word start token
            if (word_stem.endswith("*")) and (not word.startswith(word_stem[:-1])):
                is_valid = False
                break
            # Word end token
            elif (word_stem.startswith("*")) and (not word.endswith(word_stem[1:])):
                is_valid = False
                break
            # General case word stem presence and word length
            elif (word_stem not in word) or (len(word) != target_length):
                is_valid = False
                break

        if is_valid:
            return word, searched_words

    print("No valid answer retrieved.")
    return None, searched_words


def process_hints(green_letters, yellow_letters, word_length):
    """
    Converts Wordle hints into word stems usable for retrieval.

    Args:
        green_letters (list): Exact matches (green letters).
        yellow_letters (dict): Misplaced letters as {char: set(invalid_positions)}.
        word_length (int): Length of the target word.

    Returns:
        list: Word stems with positional tags (e.g., ['HE|FIRST_HALF']).
    """
    midpoint = word_length // 2
    word_stem_keys = []

    # Add green letter stems
    for pos, char in enumerate(green_letters):
        if char is not None:
            tag = "FIRST_HALF" if pos < midpoint else "SECOND_HALF"
            word_stem_keys.append(f"{char}|{tag}")

    # Add yellow letter stems, avoiding invalid positions
    for char, invalid_positions in yellow_letters.items():
        valid_positions = [i for i in range(word_length) if i not in invalid_positions]
        for pos in valid_positions:
            tag = "FIRST_HALF" if (pos < midpoint) else "SECOND_HALF"
            word_stem_keys.append(f"{char}|{tag}")

    return word_stem_keys


def find_answer(green_letters, yellow_letters, gray_letters, ground_truth, associations, searched_words=None, prob_threshold=0.001, gray_penalty=0.7, pos_penalty=0.3, start_strategy="vowels"):
    """
    Loops until the correct answer is found using the retrieve_next_valid_parallel function.

    Args:
        word_stem_keys (list): List of word stem keys that must be present in the word.
        candidate_probs (dict): Dictionary of candidate words with their probabilities.
        ground_truth (str): The correct answer to find.
        prob_threshold (float): Minimum probability threshold for a candidate to be considered; if no word has activation beyond the threshold, a random word satisfying the length constraint will be retrieved.

    Returns:
        str: The correct answer, if found, or None if no valid answer could be retrieved.
    """
    target_length = len(ground_truth)

    # Handle starting word strategies
    if (green_letters == [None] * target_length) and (not yellow_letters) and (not gray_letters):
        if (target_length == 5) and (start_strategy == "vowels"):
            start_word = random.choice(["AUDIO", "ADIEU"])
        elif (target_length == 5) and (start_strategy == "optimal"):
            start_word = random.choice(["SLATE", "CRANE", "TRACE"])
        elif (start_strategy == "random"):
            # Sample a word stem
            start_word_stem = random.choice(list(associations.keys()))
            start_word = ""
            # Sample a length-matching word
            while (len(start_word) != target_length):
                start_word = random.choice(associations[start_word_stem])["word"]
        else:
            raise ValueError("Unrecognized starting strategy.")

        print(f"Choosing starting word {start_word} under strategy '{start_strategy}'.")
        return start_word, searched_words

    # Turn hints into word stems
    word_stem_keys = process_hints(green_letters, yellow_letters, target_length)

    # Split dictionary into separate lists for words and probabilities
    candidate_probs = compute_candidate_scores(
        word_stem_keys, gray_letters, target_length, associations, sigma=1e-5, 
        gray_penalty_factor=gray_penalty, pos_penalty_factor=pos_penalty)
    words, probs = zip(*candidate_probs.items())

    # Sort words and probabilities in descending order
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    # Initialize set to track searched words
    if not searched_words:
        searched_words = set()

    while True:
        # Attempt to find the next valid word
        next_word, searched_words = retrieve_next_valid(
            word_stem_keys, target_length, sorted_words, sorted_probs, searched_words, prob_threshold
        )
        if next_word:
            print(f"Word matching all word stems found after {len(searched_words)} attempts: {next_word}")
            return next_word, searched_words

        # No valid word found; use a random length-matching word
        for i, word in enumerate(sorted_words):
            if (len(word) == target_length) and (probs[i] >= prob_threshold):
                print(f"No valid word matching all criteria found; choosing a length-matching word: {word}")
                return word, searched_words

        # No recallable word has matching target length; use random alphabetical characters
        random_string = ''.join(random.sample(string.ascii_uppercase, target_length))
        print(f"No recallable word has matching target length; using random string: {random_string}")
        return random_string, searched_words


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
    candidate_probs = compute_candidate_scores(word_stems, set(), target_length, associations)

    # Sort and select top candidates
    sorted_candidates = sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:top_n] if top_n else sorted_candidates

    return top_candidates


if __name__ == "__main__":
    with open("data/word_associations_with_pos_06.json", "r") as file:
        associations = json.load(file)

    word_stems = ["C*|FIRST_HALF", "OU|SECOND_HALF"]
    target_length = 5
    candidate_probs = compute_candidate_scores(word_stems, set(), target_length, associations)
    pprint.pprint(retrieve_top_candidates(word_stems, target_length, associations, top_n=20))
    print()

    find_answer([None, None, "O", None, None], {"L": {0}}, set("A"), "CLOUD", associations)
    print()
    find_answer([None, None, None, None, None], dict(), set(), "CLOUD", associations, start_strategy="vowels")
    