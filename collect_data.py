import json
from build_wordle import wordle_game

word_list = [
    "DROOL", "VYING", "PLUMB", "PATIO", "FLUNG",
    "ENDOW", "CRYPT", "MAUVE", "DOGMA", "HIPPO",
    "CHOCK", "SLANG", "WITCH", "BROWN", "TWIST",
    "PEARL", "SPINE", "NICHE", "GOING", "BOAST",
    "BOXER", "HYENA", "HILLY", "SHOVE", "SHAKY",
    "GUILE", "JELLY", "FRAIL", "TALLY", "VISOR",
    "TACKY", "UVULA", "PRIMP", "FLOWN", "STOIC",
    "INNER", "SWELL", "READY", "EVENT", "TRULY",
    "OCTET", "VINYL", "BLAZE", "SNOOP", "SIXTH",
    "WEIRD", "EASEL", "TUNIC", "BAWDY", "SANDY",
    "WREAK", "FROWN", "BOSSY", "GOOFY", "SHOUT",
    "CIGAR", "REBUT", "SISSY", "HUMPH", "AWAKE", 
    "BLUSH", "FOCAL", "EVADE", "NAVAL", "SERVE", 
    "HEATH", "DWARF", "MODEL", "KARMA", "STINK", 
    "GRADE", "QUIET", "BENCH", "ABATE", "FEIGN", 
    "MAJOR", "DEATH", "FRESH", "CRUST", "STOOL", 
    "COLON", "ABASE", "MARRY", "REACT", "BATTY", 
    "PRIDE", "FLOSS", "HELIX", "CROAK", "STAFF", 
    "PAPER", "UNFED", "WHELP", "TRAWL", "OUTDO", 
    "ADOBE", "CRAZY", "SOWER", "REPAY", "DIGIT", 
    "CRATE", "CLUCK", "SPIKE", "MIMIC", "POUND"
]

def collect_data(corpus_path, associations, start_strategy):
    """
    Collects model performance data for a list of target words using different starting strategies.

    Args:
        corpus_path (str): Path to the corpus file.
        associations (str): Path to the JSON file containing precomputed associations.
        start_strategy (str): Starting word strategy ("vowels", "optimal", "random").

    Returns:
        dict: Dictionary containing performance data for each word.
    """
    data = {}

    # Load associations
    with open(associations, "r") as assoc_file:
        assoc_data = json.load(assoc_file)

    # Run Wordle game for each word
    for word in word_list:
        guess_list, finished_game = wordle_game(
            corpus_path, ground_truth=word, attempt_limit=6, auto=True,
            prob_threshold=0.001, valid_threshold=1e-6, pos_penalty=0.3,
            associations=assoc_data, start_strategy=start_strategy
        )

        # Process data into appropriate structure
        data[word] = {
            "guesses": guess_list,
            "success": finished_game
        }

    return data


associations_list = ["00", "02", "04", "06", "08"]
start_strategy_list = ["vowels", "optimal", "popular", "random"]
corpus_path = "data/thorndike_corpus.csv"


if __name__ == "__main__":

    # Collect and store results
    all_results = {}
    for assoc in associations_list:
        for start_strategy in start_strategy_list:
            key = f"associations{assoc}_{start_strategy}"
            print(f"Processing: {key}")
            all_results[key] = collect_data(
                corpus_path, f"data/word_associations_{assoc}.json", start_strategy)

    # Save results to a JSON file
    with open("results/results.json", "w") as file:
        json.dump(all_results, file, indent=4)
