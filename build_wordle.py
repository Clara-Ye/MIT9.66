import random
import json
import pandas as pd

from learn_words import load_corpus, extract_word_stems
from retrieve import find_answer


def filter_corpus(corpus_path, min_length, max_length, min_freq):
    """
    Filters the corpus to include only words within the specified length range.

    Args:
        corpus (list): List of words.
        min_length (int): Minimum word length.
        max_length (int): Maximum word length.

    Returns:
        list: Filtered list of words.
    """
    corpus = load_corpus(corpus_path)
    return [
            row["Word"] for _, row in corpus.iterrows()
            if (min_length <= len(row["Word"]) <= max_length) and (row["Frequency"] >= min_freq)
        ]


def wordle_game(corpus_path, ground_truth=None, attempt_limit=6, min_word_length=4, max_word_length=7, min_frequency=10, associations=None, prob_threshold=0.001):
    """
    Simulates a Wordle-like game.

    Args:
        corpus (list): List of words to sample from.
        attempt_limit (int): Maximum number of attempts allowed.
        min_word_length (int): Minimum word length for the target word.
        max_word_length (int): Maximum word length for the target word.
        associations (dict): Word-stem associations.
    """
    # Sample a random word from the filtered corpus if ground truth is not provided
    if not ground_truth:
        corpus = filter_corpus(corpus_path, min_word_length, max_word_length, min_frequency)
        ground_truth = random.choice(corpus)
        
    target_length = len(ground_truth)
    print(f"The target word has {target_length} letters. You have {attempt_limit} attempts.")

    attempts = 0
    green_letters = [None for _ in range(target_length)]
    searched_words = set()
    while attempts < attempt_limit:
        attempts += 1
        print(f"\nAttempt {attempts} of {attempt_limit}")

        # Get input from user or model
        guess = input(f"Enter a {target_length}-letter word or type 'ENTER' for model suggestion: ").strip()
        if guess == "":
            if associations:
                guess, searched_words = find_answer(
                    green_letters, ground_truth, associations,
                    searched_words=searched_words,
                    prob_threshold=prob_threshold)
                prob_threshold /= 2
            else:
                print("No model associations provided. User inputs only.")
                continue

        # Validate guess length
        if len(guess) != target_length:
            print(f"Invalid input. Please enter a {target_length}-letter word.")
            attempts -= 1
            continue

        # Keep track of user input guessed word
        searched_words.add(guess.upper())

        # Check if the guessed word matches the ground truth
        if guess.upper() == ground_truth.upper():
            print(f"Congratulations! You guessed the correct word: {ground_truth}\n")
            return

        # Check for exact matches and generate positional information
        new_green_letters = [g if g == t else None for g, t in zip(guess.upper(), ground_truth.upper())]
        print(f"Exact matches: {new_green_letters}")        
        green_letters = [green_letters[i] if green_letters[i] is not None else new_green_letters[i] for i in range(target_length)]
        print(f"All past exact matches: {green_letters}")

        # Generate and display matching word stems
        ground_truth_stems = extract_word_stems(ground_truth)
        guessed_word_stems = extract_word_stems(guess)
        matching_stems = list(set(ground_truth_stems) & set(guessed_word_stems))
        print(f"Matching word stems: {matching_stems}")

    # Reveal the correct answer if attempts are exhausted
    print(f"Sorry, you've used all {attempt_limit} attempts. The correct word was: {ground_truth}\n")


if __name__ == "__main__":
    # Load the corpus and associations
    corpus_path = "data/thorndike_corpus.csv"

    with open("data/word_associations_with_pos_06.json", "r") as f:
        associations = json.load(f)

    # Run the Wordle game
    wordle_game(corpus_path, ground_truth="CLOUD", associations=associations, prob_threshold=0.001)
    wordle_game(corpus_path, ground_truth="WAGON", associations=associations, prob_threshold=0.001)
    #wordle_game(corpus_path, associations=associations, prob_threshold=0.01)
