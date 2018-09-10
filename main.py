"""Process concreteness data from MRC and identity minimal sets.

Then estimate impact of particular onsets (or other components) on concreteness."""

import numpy as np
import pandas as pd 
from nlp_utilities.compling import CorpusUtilities

from src.utils import get_concreteness_for_word


# Set global variables
DATA_PATH = "data/raw/mrc_concreteness.csv"
CHARACTER_INDEX	= 0

# Get data
df = pd.read_csv(DATA_PATH, sep="\t")
df['word'] = [w.lower() for w in df.Word]

# Get minimal sets
minimal_sets = CorpusUtilities.get_minimal_orthographic_sets(df['word'], index=CHARACTER_INDEX)



COLUMNS = ['letter', 'concreteness_impact', 'set_mean', 'original_word', 'set_index']

letter_to_impact = []
for set_index, ms in enumerate(minimal_sets):
	letters = [w[CHARACTER_INDEX] for w in ms]
	by_word_cnc = get_concreteness_for_word(ms, df)
	mean_cnc = np.mean(by_word_cnc)
	for index, letter in enumerate(letters):
		to_add = [letter, by_word_cnc[index] - mean_cnc, mean_cnc, ms[index], set_index]
		letter_to_impact.append(to_add)

final_df = pd.DataFrame(letter_to_impact, columns=COLUMNS)

final_df.to_csv("data/processed/concreteness_impacts.csv")