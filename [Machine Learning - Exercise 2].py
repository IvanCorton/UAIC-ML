import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import math



X = pd.DataFrame({'X1': [1, 1, 1, 1, 0, 0],
                  'X2': [1, 1, 1, 0, 0, 0]})
YSeries = pd.Series([1, 1, 2, 3, 2, 3])


X1 = np.array([1, 1, 1, 1, 0, 0])
X2 = np.array([1, 1, 1, 0, 0, 0])
Y = np.array([1, 1, 2, 3, 2, 3])


##Entropy
def entropy(Y):
    """
    Entropy = Sum( -[prob of 0's & 1's] * log2 [prob 0's & 1's])
    """
    discrimination, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    print(prob)
    return en


#Joint Entropy
def jEntropy(Y,X):
    """
    Joint Entropy = H(Y;X)
    """
    YX = np.c_[Y,X]         #Joining attributes from X and Y in a column.
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    Conditional Entropy = H(Y|X) = H(Y;X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)


#Information Gain
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y,X)



print("Information gained using X1: ", gain(Y,X1))
print("Information gained using X2: ",gain(Y,X2))


# I think the best option to use in the ID3 algorithm would be
# the one that provides more information (and less entropy), so, taking 
# a look into the results, the atributte X1 will be better for our algorithm (Better IG and less H).

X1 = np.matrix(X[['X1']])
X2 = np.matrix(X[['X2']])

# Classify of 'X1' : 0

cat_features = X1 # Note that we're selecting a matrix
enc = OneHotEncoder(sparse=False).fit(cat_features)
X_transformed = pd.DataFrame(enc.transform(cat_features), columns=enc.categories_)
X_transformed


from sklearn import tree
import matplotlib.pyplot as plt
y = Y
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X_transformed,y)
fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X_transformed.columns)
plt.show()


# Classify of 'X2' : 1

cat_features2 = X2 # Note that we're selecting a matrix
enc = OneHotEncoder(sparse=False).fit(cat_features2)
X2_transformed = pd.DataFrame(enc.transform(cat_features2), columns=enc.categories_)
X2_transformed


from sklearn import tree
import matplotlib.pyplot as plt
y = Y
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X2_transformed,y)
fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X2_transformed.columns)
plt.show()