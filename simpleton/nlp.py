import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def count_tokens(column: pd.core.series.Series, ngrams: int) -> pd.core.frame.DataFrame:
    """
    Consumes a pandas dataframe column and counts the number of tokens.

    Parameters
    ----------
    column: pd.core.series.Series
         Input list
    ngrams: int
        The number of tokens 1 = unigram, 2 = bigram, 3 = trigram, etc.
    Returns
    -------
    output: pd.core.frame.DataFrame
        A dataframe containing the token and count.
    """
    word_vectorizer = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(column)
    frequencies = sum(sparse_matrix).toarray()[0]

    output = pd.DataFrame(
        frequencies,
        index=word_vectorizer.get_feature_names(),
        columns=['Frequency']).sort_values(
        "Frequency",
        ascending=False)
    output.reset_index(level=0, inplace=True)
    output.rename(columns={'index': 'Token'}, inplace=True)

    return output
