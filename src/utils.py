"""Utility functions."""

def get_concreteness_for_word(words, df, column='word'):
	"""Look up concreteness of each word in dataframe"""
	cnc = []
	for word in words:
		cnc.append(float(df[df[column] == word]['Concreteness']))
	return cnc

