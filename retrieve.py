import json
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


def compute_candidate_scores(word_stem_keys, target_length, associations, sigma=1e-5, pos_penalty=0.3):
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
                
                # Check rough positional alignment
                if (rough_position == "FIRST_HALF" and word.find(word_stem) <= len(word) // 2) or \
                (rough_position == "SECOND_HALF" and word.find(word_stem) > len(word) // 2):
                    candidate_probs[word] = candidate_probs.get(word, 1) * (entry["prob"] * pos_penalty)
                else:
                    candidate_probs[word] = candidate_probs.get(word, 1) * (entry["prob"])
    else:
        # If no word stems, consider all words in associations
        for word_stem_key, assoc in associations.items():
            for entry in assoc:
                word, prob = entry["word"], entry["prob"]
                candidate_probs[word] = prob

    # Apply smoothing and normalize by number of stems
    n_stems = max(1, len(word_stem_keys))  # Avoid division by zero
    candidate_probs = {
        word: (prob + sigma) ** (1 / n_stems) for word, prob in candidate_probs.items()
    }

    # Adjust by length and normalize
    adjust_probabilities_by_length(candidate_probs, target_length)
    normalize(candidate_probs)

    return candidate_probs


def retrieve_next_valid(green_letters, yellow_letters, gray_letters, target_length, words, probs, searched_words, prob_threshold=0.0, valid_threshold=1e-6):
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
        if (word in searched_words) or (prob < prob_threshold):
            continue

        searched_words.add(word)

        # Ensure valid word length
        if (len(word) != target_length):
            continue

        # Check if the word matches all hints
        validity = 1.0
        # Green letters
        for i, char in enumerate(green_letters):
            # Valid to try out more different characters
            if (char is not None) and (char not in word):
                validity *= 0.4
            # Invalid to ignore the hint
            elif (char is not None) and (word[i] != char):
                validity *= 0.1

        # Yellow letters
        for char, invalid_pos in yellow_letters.items():
            # The letter should be present
            if (char not in word):
                validity *= 0.2
            # The letter should not appear in a recorded wrong position
            for i in invalid_pos:
                if (word[i] == char):
                    validity *= 0.1

        # Gray letters
        for char in gray_letters:
            # The letter should not appear in the word
            if (char in word):
                validity *= 0.1

        # Select the word based on validity
        if (validity > valid_threshold) and (np.random.random() < validity):
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


def generate_random_word(green_letters, yellow_letters, gray_letters, target_length):
    """
    Generates a random word avoiding gray letters and invalid yellow positions.

    Args:
        green_letters (list): Exact matches (green hints) with positions.
        yellow_letters (dict): Misplaced letters as {char: set(invalid_positions)}.
        gray_letters (set): Letters not in the target word (gray hints).
        target_length (int): Length of the target word.

    Returns:
        str: A randomly generated word following the constraints.
    """
    valid_chars = []

    # Create a pool of valid characters for each position
    for i in range(target_length):
        # Start with all alphabetic characters
        pool = set(string.ascii_uppercase)

        # Exclude green and gray letters
        pool -= set(green_letters)
        pool -= gray_letters

        # Exclude yellow letters in invalid positions
        for char, invalid_positions in yellow_letters.items():
            if (i in invalid_positions) and (char in pool):
                pool.remove(char)

        # Add the filtered pool for this position
        valid_chars.append(pool)

    # Generate the random word
    random_word = ''.join(np.random.choice(list(chars)) for chars in valid_chars)

    return random_word


def find_answer(green_letters, yellow_letters, gray_letters, ground_truth, associations, searched_words=None, prob_threshold=0.001, valid_threshold=1e-6, pos_penalty=0.3, start_strategy="vowels"):
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
            start_word = np.random.choice(["AUDIO", "ADIEU"])
        elif (target_length == 5) and (start_strategy == "optimal"):
            start_word = np.random.choice(["SLATE", "CRANE", "TRACE"])
        elif (start_strategy == "popular"):
            start_word = np.random.choice([
                "STARE", "RAISE", "ARISE", "IRATE", "TRAIN", "GREAT",
                "HEART", "AROSE", "HOUSE", "AISLE", "STEAM", "LEAST",
                "CRATE", "TEARS", "SALET", "DREAM"],
                p=[0.215, 0.155, 0.09, 0.07, 0.07, 0.045, 0.045, 0.04,
                   0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03])
        elif (start_strategy == "random"):
            # Flatten the dictionary into a list of all word objects
            word_pool = [word_obj for entry in associations.values() for word_obj in entry]
            # Filter the pool for words with the target length
            word_pool = [word_obj for word_obj in word_pool if len(word_obj["word"]) == target_length]
            # Sample a random word
            start_word = np.random.choice(word_pool)["word"]
        else:
            raise ValueError("Unrecognized starting strategy.")

        print(f"Choosing starting word {start_word} under strategy '{start_strategy}'.")
        return start_word, searched_words

    # Turn hints into word stems
    word_stem_keys = process_hints(green_letters, yellow_letters, target_length)

    # Split dictionary into separate lists for words and probabilities
    candidate_probs = compute_candidate_scores(
        word_stem_keys, target_length, associations, pos_penalty=pos_penalty)
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
            green_letters, yellow_letters, gray_letters, target_length,
            sorted_words, sorted_probs, searched_words, prob_threshold,
            valid_threshold=valid_threshold)
        if next_word:
            print(f"Word matching most of the hints found after {len(searched_words)} attempts: {next_word}")
            return next_word, searched_words

        # No reasonable word found; use random alphabetical characters
        random_string = generate_random_word(green_letters, yellow_letters, gray_letters, target_length)
        print(f"No reasonable word given the hints found; using random string: {random_string}")
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
    candidate_probs = compute_candidate_scores(word_stems, target_length, associations)

    # Sort and select top candidates
    sorted_candidates = sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_candidates[:top_n] if top_n else sorted_candidates

    return top_candidates


if __name__ == "__main__":
    with open("data/word_associations_with_pos_06.json", "r") as file:
        associations = json.load(file)

    word_stems = ["C*|FIRST_HALF", "OU|SECOND_HALF"]
    target_length = 5
    candidate_probs = compute_candidate_scores(word_stems, target_length, associations)
    pprint.pprint(retrieve_top_candidates(word_stems, target_length, associations, top_n=20))
    print()

    find_answer([None, None, "O", None, None], {"L": {0}}, set("A"), "CLOUD", associations)
    print()
    find_answer([None, None, None, None, None], dict(), set(), "CLOUD", associations, start_strategy="popular")
    