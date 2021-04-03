import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def apply_counts(df: pd.DataFrame, count_col: str):
    """ Denormalise a dataframe with a 'Counts' column by
    multiplying that column by the count and dropping the 
    count_col. """
    feats = [c for c in df.columns if c != count_col]
    return pd.concat([
        pd.DataFrame([list(r[feats])] * r[count_col], columns=feats)
        for i, r in df.iterrows()
    ], ignore_index=True)


d = pd.DataFrame({'X1': [0, 0, 1, 1, 0, 0, 1, 1],
                  'X2': [0, 0, 0, 0, 1, 1, 1, 1],
                  'C' : [2, 18, 4, 1, 4, 1, 2, 18],
                  'Y' : [0, 1, 0, 1, 0, 1, 0, 1]})
d=apply_counts(d, 'C')

e = d[['X1', 'X2']]

Y = d['Y']

Xsample = pd.DataFrame({'X1':[0], 
                        'X2':[0]})

print(Xsample)

cl = BernoulliNB()
cl.fit(e, Y)

cl.predict(Xsample)

cl.predict_proba(Xsample)

nb = MultinomialNB().fit(e, Y)
print(nb.class_log_prior_)      # Produces 2 results (2 features + 1 label => 2 prob.)          n label possibilities [0 or 1] = m prob.

print(nb.feature_log_prob_)     # Produces a matrix of 2 arrays of results (n features * label possibilities [0 or 1]) = m prob. 



