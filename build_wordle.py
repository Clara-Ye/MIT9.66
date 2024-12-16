import random
import json
from learn_words import load_corpus
from retrieve import find_answer


def filter_corpus(corpus_path, min_word_length, max_word_length, min_frequency):
    """
    Filters the corpus to include only words within the specified length range and frequency threshold.

    Args:
        corpus_path (str): Path to the corpus file.
        min_length (int): Minimum word length.
        max_length (int): Maximum word length.
        min_freq (int): Minimum frequency threshold.

    Returns:
        list: Filtered list of words.
    """
    corpus = load_corpus(corpus_path)
    return [
        row["Word"] for _, row in corpus.iterrows()
        if (min_word_length <= len(row["Word"]) <= max_word_length) and \
        (row["Frequency"] >= min_frequency)
    ]

def wordle_game(corpus_path, ground_truth=None, attempt_limit=6, 
                min_word_length=4, max_word_length=6,
                min_frequency=10, associations=None,
                prob_threshold=0.001, valid_threshold=1e-6,
                pos_penalty=0.3, start_strategy="vowels", auto=True):
    """
    Simulates a Wordle-like game with feedback for green (exact), yellow (present but misplaced), and gray (absent) matches.
    Includes an auto mode for fully automated gameplay.

    Args:
        corpus_path (str): Path to the corpus file.
        ground_truth (str): Target word (if None, a random word from the corpus is chosen).
        attempt_limit (int): Maximum number of allowed attempts.
        min_word_length (int): Minimum word length for filtering the corpus.
        max_word_length (int): Maximum word length for filtering the corpus.
        min_frequency (int): Minimum frequency for filtering the corpus.
        associations (dict): Precomputed word-stem associations of the model.
        prob_threshold (float): Minimum probability threshold for a word to be considered by the model.
        valid_threshold (float): Threshold for word validity during probabilistic selection.
        pos_penalty (float): Penalty factor for mismatched positional constraints.
        start_strategy (str): Strategy for selecting the starting word ("vowels", "optimality", "random").
        auto (bool): Whether the game should run in auto mode.

    Returns:
        list: List of guesses made by the model.
    """
    if not ground_truth:
        corpus = filter_corpus(corpus_path, min_word_length, max_word_length, min_frequency)
        ground_truth = random.choice(corpus)

    target_length = len(ground_truth)
    print(f"The target word has {target_length} letters. You have {attempt_limit} attempts.")

    attempts = 0
    green_letters = [None] * target_length  # Exact matches
    yellow_letters = {}  # Misplaced letters: {char: set(invalid_positions)}
    gray_letters = set()  # Letters not in the target word
    searched_words = set()
    guess_list = []

    while attempts < attempt_limit:
        attempts += 1
        print(f"\nAttempt {attempts} of {attempt_limit}")

        if auto:
            # Get model's suggestion
            if associations:
                guess, searched_words = find_answer(
                    green_letters, yellow_letters, gray_letters, ground_truth,
                    associations, searched_words, prob_threshold, valid_threshold,
                    pos_penalty, start_strategy)
                prob_threshold /= 2  # Adjust threshold for subsequent guesses
            else:
                print("No model associations provided for auto mode. Terminating.")
                return guess_list
        else:
            # Get user input
            guess = input(f"Enter a {target_length}-letter word or type 'ENTER' for model suggestion: ").strip().upper()
            if not guess:
                if associations:
                    guess, searched_words = find_answer(
                        green_letters, yellow_letters, gray_letters, ground_truth,
                        associations, searched_words, prob_threshold, valid_threshold,
                        pos_penalty, start_strategy)
                    prob_threshold /= 2
                else:
                    print("No model associations provided. Please input manually.")
                    attempts -= 1
                    continue

        # Validate guess length
        if (len(guess) != target_length):
            print(f"Invalid input. Please enter a {target_length}-letter word.")
            attempts -= 1
            continue

        # Record guess to avoid repeating attempts
        searched_words.add(guess)
        guess_list.append(guess)

        # Check if the guess is correct
        if (guess == ground_truth):
            print(f"Congratulations! You guessed the correct word: {ground_truth}\n")
            return guess_list, True

        # Check for exact matches
        new_green_letters = [
            g if (g == t) else None for g, t in zip(guess, ground_truth)]
        green_letters = [
            old or new for old, new in zip(green_letters, new_green_letters)]
        #print(f"Exact matches (green): {green_letters}")

        # Check for misplaced (yellow) and missed letters (gray)
        for i, char in enumerate(guess):
            if (char in ground_truth) and (char != ground_truth[i]):
                yellow_letters.setdefault(char, set()).add(i)
            elif (char not in ground_truth):
                gray_letters.add(char)
        #print(f"Yellow letters (misplaced): {yellow_letters}")
        #print(f"Gray letters (not in word): {gray_letters}")

    # Reveal the answer if all attempts are exhausted
    print(f"Sorry, you've used all {attempt_limit} attempts. The correct word was: {ground_truth}\n")

    return guess_list, False


if __name__ == "__main__":
    # Load the corpus and associations
    corpus_path = "data/thorndike_corpus.csv"
    with open("data/word_associations_with_pos_06.json", "r") as f:
        associations = json.load(f)

    # Run Wordle games with example ground truth
    print(wordle_game(corpus_path, ground_truth="DROOL", associations=associations, prob_threshold=0.001))
    print()
    print(wordle_game(corpus_path, ground_truth="VYING", associations=associations, prob_threshold=0.001))
    print()
    #wordle_game(corpus_path, associations=associations, prob_threshold=0.01, start_strategy="random")
