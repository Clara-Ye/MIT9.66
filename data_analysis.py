import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load model data
with open('results/results.json', 'r') as file:
    model_data = json.load(file)

# Load human data
with open('data/wordlebot_data.json', 'r') as file:
    human_data = json.load(file)


def visualize(model_names, variable):
    # Initialize dictionaries to store results
    success_rates = {}
    average_turns = {}

    for model in model_names:
        curr_model_data = model_data[model]

        total_games = len(curr_model_data)
        successful_games = sum(1 for game in curr_model_data.values() if game['success'])

        # Calculate success rate
        success_rate = (successful_games / total_games) * 100
        success_rates[model] = success_rate

        # Calculate average number of turns for successful games
        turns = [len(game["guesses"]) for game in curr_model_data.values() if game["success"]]
        average_turns[model] = np.mean(turns) if turns else 0

    # Plotting success rates
    plt.figure(figsize=(7, 4))
    bars = plt.barh(list(success_rates.keys()), list(success_rates.values()), alpha=0.8)
    for bar in bars:
        plt.text(bar.get_width() - 8, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va="center", color="white")
    plt.title(f"Success Rates Across {variable.capitalize()}")
    plt.xlabel("Success Rate (%)")
    plt.tight_layout()
    plt.savefig(f"results/success_rates_vs_{variable}.png")

    # Plotting average turns
    plt.figure(figsize=(7, 4))
    bars = plt.barh(list(average_turns.keys()), list(average_turns.values()), alpha=0.8)
    for bar in bars:
        plt.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va="center", color="white")
    plt.title(f"Average Turns Across {variable.capitalize()}")
    plt.xlabel("Average Turns")
    plt.tight_layout()
    plt.savefig(f"results/n_turns_vs_{variable}.png")


def create_summary_table(data):
    summary = []

    for model, model_data in data.items():
        association_level, strategy = model.split('_', 1)

        total_games = len(model_data)
        successful_games = sum(1 for game in model_data.values() if game['success'])

        # Calculate success rate
        success_rate = (successful_games / total_games) * 100

        # Calculate average number of turns for successful games
        turns = [len(game['guesses']) for game in model_data.values() if game['success']]
        avg_turns = np.mean(turns) if turns else 0

        summary.append({
            "Association Level": association_level,
            "Strategy": strategy,
            "Success Rate (%)": success_rate,
            "Average Turns": avg_turns
        })

    # Convert to DataFrame for better visualization
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(by=["Association Level", "Strategy"])
    summary_df.to_csv("results/summary_table.csv", index=False)


def compare_with_human_data(model_data, human_data):
    comparison = []

    for model, guess_data in model_data.items():
        for word, details in guess_data.items():
            if word in human_data:
                guesses_in_human_top_10 = 0

                # Check guesses against human top 10 per turn
                for i, guess in enumerate(details["guesses"]):
                    human_top_10 = [entry["word"] for entry in human_data[word]["turn_data"].get(f"{i+1}", [])]

                    if guess in human_top_10:
                        guesses_in_human_top_10 += 1

                model_turns = len(details["guesses"])
                human_turns = human_data[word].get("average_turns", None)
                percentage_overlap = guesses_in_human_top_10 / model_turns

                comparison.append({
                    "Model": model,
                    "Word": word,
                    "Model Turns": model_turns,
                    "Human Turns": human_turns,
                    "% in Human Top 10": percentage_overlap
                })

    # Convert to DataFrame for better visualization
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values(by=["Model", "Word"])
    comparison_df.to_csv("results/comparison_with_human.csv", index=False)


if __name__ == "__main__":

    visualize(["associations00_vowels", "associations00_optimal",
               "associations00_popular", "associations00_random"],
               variable="strategies (eta=0.0)")

    visualize(["associations02_vowels", "associations02_optimal",
               "associations02_popular", "associations02_random"],
               variable="strategies (eta=0.2)")

    visualize(["associations04_vowels", "associations04_optimal",
               "associations04_popular", "associations04_random"],
               variable="strategies (eta=0.4)")

    visualize(["associations06_vowels", "associations06_optimal",
               "associations06_popular", "associations06_random"],
               variable="strategies (eta=0.6)")

    visualize(["associations08_vowels", "associations08_optimal",
               "associations08_popular", "associations08_random"],
               variable="strategies (eta=0.8)")

    visualize(["associations00_vowels", "associations02_vowels",
               "associations04_vowels", "associations06_vowels",
               "associations08_vowels"],
               variable="associations (vowels)")

    visualize(["associations00_optimal", "associations02_optimal",
               "associations04_optimal", "associations06_optimal",
               "associations08_optimal"],
               variable="associations (optimal)")

    visualize(["associations00_popular", "associations02_popular",
               "associations04_popular", "associations06_popular",
               "associations08_popular"],
               variable="associations (popular)")

    visualize(["associations00_random", "associations02_random",
               "associations04_random", "associations06_random",
               "associations08_random"],
               variable="associations (random)")

    create_summary_table(model_data)
    compare_with_human_data(model_data, human_data)
