"""Generate nonwords beginning with specific onsets.

Info about phonDISC: 
http://groups.linguistics.northwestern.edu/speech_comm_group/documents/CELEX/Phonetic%20codes%20for%20CELEX.pdf
"""

import pandas as pd 
import editdistance as ed

from collections import Counter
from nlp_utilities.compling import CorpusUtilities
from tqdm import tqdm


# Read in CELEX
df_celex = pd.read_csv("data/raw/celex_all.csv", sep = "\\")
df_celex = df_celex.drop_duplicates(subset='Word')
print("Number of words in CELEX: {num}".format(num=len(df_celex)))


## Get nonwords
# Filter on monosyllables
df_mono = df_celex[df_celex['SylCnt']==1]
df_mono = df_mono.reset_index()
print("Number of monosyllabic words: {num}".format(num=len(df_mono)))

df_mono['Rime'] = df_mono['PhonDISC'].apply(lambda x: CorpusUtilities.get_rime(x))
df_mono = df_mono.dropna()

# Get frequencies
rime_freqs = Counter(df_mono['Rime'])


###### Make new words
def get_example(rime, df_og):
	return df_og[df_og['Rime']==rime].sample(1)['Word'].iloc[0]

##### Get phonological neighborhood data
def get_phonological_neighborhood_stats(phonetic_transcription, mrc, celex):
	"""Get stats about phonological neighborhood."""
	celex['distance'] = celex['PhonDISC'].apply(lambda x: ed.eval(phonetic_transcription, x))
	celex_neighbors = celex[celex['distance'] <= 1]
	words = list(celex_neighbors['Word'])

	mrc_neighbors = mrc[mrc['Word'].isin(words)]

	return {'mean_frequency_neighbors': celex_neighbors['CobLog'].mean(),
			'sd_frequency_neighbors': celex_neighbors['CobLog'].std(),
			'mean_concreteness_neighbors': mrc_neighbors['Concreteness'].mean(),
			'sd_concreteness_neighbors': mrc_neighbors['Concreteness'].std(),
			'neighborhood_size_celex': len(words),
			'neighborhood_size_mrc': len(mrc_neighbors),
			'celex_neighbors': words
			}


## Get MRC data
df_mrc = pd.read_csv("data/raw/mrc_concreteness.csv", sep="\t")
df_mrc['Word'] = df_mrc['Word'].str.lower()

## Target onsets
TARGETS = ['D', #th
		   'J' #ch
		   ]

## Generate words
nonwords = []
for rime, freq in tqdm(rime_freqs.items()):
	example = get_example(rime, df_mono)
	temps = []

	for onset in TARGETS:
		temp = onset + rime
		if temp not in list(df_celex['PhonDISC']):
			temps.append(temp)

	if len(temps) == len(TARGETS):
		for w in temps:
			stats = get_phonological_neighborhood_stats(w, df_mrc, df_celex)
			nonwords.append({
				'nonword': w,
				'rime': rime,
				'rime_freq': freq,
				'example_word': example,
				'mean_frequency_neighbors': stats['mean_frequency_neighbors'],
				'sd_frequency_neighbors': stats['sd_frequency_neighbors'],
				'mean_concreteness_neighbors': stats['mean_concreteness_neighbors'],
				'sd_concreteness_neighbors': stats['sd_concreteness_neighbors'],
				'neighborhood_size_celex': stats['neighborhood_size_celex'],
				'neighborhood_size_mrc': stats['neighborhood_size_mrc'],
				'celex_neighbors': stats['celex_neighbors']
				})

df_rime = pd.DataFrame(nonwords) 
df_rime.to_csv("data/processed/nonwords.csv")