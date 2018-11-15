"""Utility functions."""


def get_concreteness_for_word(words, df, column='word'):
	"""Look up concreteness of each word in dataframe"""
	cnc = []
	for word in words:
		cnc.append(float(df[df[column] == word]['Concreteness']))
	return cnc

def get_vector_for_word(words, df, column='Word', value="word2vec"):
	"""Look up concreteness of each word in dataframe"""
	vecs = []
	for word in words:
		vec = df[df[column] == word]['word2vec'].iloc[0].replace("[", "").replace("]", "").strip()
		vec = [float(dim) for dim in vec.split()]
		vecs.append(vec)
	return vecs