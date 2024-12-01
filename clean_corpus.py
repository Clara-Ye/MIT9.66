import pandas as pd
import re

data = []

with open(r"data\thorndike_corpus.txt", 'r') as file:
    for line in file:

        # read frequency
        try:
            frequency = int(line[:6].replace(" ", "0")) # some frequencies contain spaces, like 0000 4
        except Exception as e:
            print(e)
            print(line)
            continue

        # read word
        word = line[7:].replace("$", "") # capitalized words start with $
        # Remove everything after the first "("
        pattern = re.compile(r"\(.*")
        word = pattern.sub("", word).strip()
        """
        match = pattern.search(word)
        if match: print(f"Matched: {match.group(0)} from word: {word}")
        """
        if all(char not in word for char in [".", ",", "'", " "]):
            data.append((word, frequency))
        
df = pd.DataFrame(data, columns=["Word", "Frequency"])
print(df.head(20))
print(len(df))
df.to_csv("data/thorndike_corpus.csv", index=False)
