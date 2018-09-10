"""Utility functions."""

def get_concreteness_for_word(words, df):
	"""Look up concreteness of each word in dataframe"""
	cnc = []
	for word in words:
		cnc.append(float(df[df['word'] == word]['Concreteness']))
	return cnc

