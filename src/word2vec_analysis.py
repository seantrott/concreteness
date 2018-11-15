"""Analyze impact of onset on word2vec dimensions."""

import json
import os

import pandas as pd
import numpy as np


import statsmodels.formula.api as sm

from ast import literal_eval
from gensim.models import KeyedVectors
from tqdm import tqdm

from nlp_utilities.compling import CorpusUtilities
from utils import get_vector_for_word


LOAD_MODEL = False
BUILD_SETS = False

# Load data
if LOAD_MODEL:
	DATA_PATH = "data/raw/celex_all.csv"
	df = pd.read_csv(DATA_PATH, sep="\\")
	df = df.drop_duplicates(subset="PhonDISC")

	# Load word2vec model
	print("Loading model...")
	MODEL_PATH = os.environ['WORD2VEC_PATH']
	model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True) 

	# Take only words from word2vec model
	print("Using only words from word2vec model...")
	df = df[df['Word'].isin(model.vocab)]
	print("Number of words left: {num}".format(num=len(df)))

	df['word2vec'] = df['Word'].apply(lambda x: model[x])
	df.to_csv("data/processed/celex_with_word2vec.csv")
else:
	print("Loading stored data...")
	df = pd.read_csv("data/processed/celex_with_word2vec.csv")
	print("Number of words: {num}".format(num=len(df)))

## get minimal sets
CHARACTER_INDEX = 0
if BUILD_SETS:
	# For now, restrict to monomorphemic words
	df = df[df['CompCnt']==1]
	print("Number of words: {num}".format(num=len(df)))
	minimal_sets = CorpusUtilities.get_minimal_orthographic_sets(df['PhonDISC'], index=CHARACTER_INDEX)
	with open("data/processed/word2vec_minimal_sets.json", "w") as outfile:
		json.dump(minimal_sets, outfile)
	# Write to file
else:
	minimal_sets = json.load(open("data/processed/word2vec_minimal_sets.json"))


# 
# Calculate onset impact
onset_to_impact = []
for set_index, ms in tqdm(list(enumerate(minimal_sets))):
	letters = [w[CHARACTER_INDEX] for w in ms]
	by_word_cnc = get_vector_for_word(ms, df, column='PhonDISC')
	mean_embedding = np.array(by_word_cnc).mean(axis=0)
	for index, onset in enumerate(letters):
		difference_vector = by_word_cnc[index] - mean_embedding
		to_add = [onset, difference_vector, mean_cnc, ms[index], set_index]
		onset_to_impact.append(to_add)

COLUMNS = ['onset', 'difference_vector', 'set_mean_embedding', 'original_word', 'set_index']
final_df = pd.DataFrame(onset_to_impact, columns=COLUMNS)


### Dimensionality reduction
onsets = list(final_df['onset'])
X = list(final_df['difference_vector'])

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)


# First visualization

import seaborn as sn 

dim1 = [i[0] for i in X_reduced]
new_df = pd.DataFrame.from_dict({'onset': onsets, 'dimension1': dim1})

ax = sn.boxplot(data=new_df, x="onset", y="dimension1")
ax.set(xlabel="Onset phone", ylabel="Impact in dimension 1")
ax.axhline(y=0, linestyle="dotted")


result = sm.ols(formula="dimension1 ~ onset", data=new_df).fit()
