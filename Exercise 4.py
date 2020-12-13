from sklearn import datasets as ds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d


c = ListedColormap(['red', 'green'])

np.random.seed(0)
X1, _ = ds.make_circles(n_samples=1500, factor=.5,  noise=.05)
X2, _ = ds.make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5], random_state=170)

d1 = pd.DataFrame(X1, columns=['X1', 'X2'])
d2 = pd.DataFrame(X2, columns=['X1', 'X2'])


# --------------------------------
# Plot_dendrogram from tools.plots
# --------------------------------

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# ----------------------------------------------------
# Different linkages to X1 (single, complete, average)
# ----------------------------------------------------

for linkage in ['single', 'complete', 'average']:

    ac = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    clusters = ac.fit_predict(X1)

    plt.scatter(d1['X1'], d1['X2'], c=clusters, cmap=c)
    plt.title(f"{linkage} linkage X1:")
    plt.show()

    # setting distance_threshold=0 ensures we compute the FULL TREE.
    ap = AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None)
    ap = ap.fit(X1)

    plot_dendrogram(ap, truncate_mode='level', p=1)
    plt.title(f"{linkage} linkage X1")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

# ----------------------------------------------------
# Different linkages to X2 (single, complete, average)
# ----------------------------------------------------

    ac = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    clusters = ac.fit_predict(X2)

    plt.scatter(d2['X1'], d2['X2'], c=clusters, cmap=c)
    plt.title(f"{linkage} linkage X2:")
    plt.show()

    # setting distance_threshold=0 ensures we compute the FULL TREE.
    ax = AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None)
    ax = ax.fit(X2)

    plot_dendrogram(ax, truncate_mode='level', p=1)
    plt.title(f"{linkage} linkage X2")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


"""
ac = AgglomerativeClustering(linkage='single', n_clusters=2)
clusters = ac.fit_predict(X1)

plt.scatter(d1['X1'], d1['X2'], c=clusters, cmap=c)
plt.title(f"Single linkage X1:")
plt.show()

ac = AgglomerativeClustering(linkage='single', n_clusters=2)
clusters = ac.fit_predict(X2)

plt.scatter(d2['X1'], d2['X2'], c=clusters, cmap=c)
plt.title(f"Single linkage X2:")
plt.show()


# ----------------
# Complete Linkage.
# ----------------

ac = AgglomerativeClustering(linkage='single', n_clusters=2)
clusters = ac.fit_predict(X1)

plt.scatter(d1['X1'], d1['X2'], c=clusters, cmap=c)
plt.title(f"Single linkage X1:")
plt.show()

ac = AgglomerativeClustering(linkage='single', n_clusters=2)
clusters = ac.fit_predict(X2)

plt.scatter(d2['X1'], d2['X2'], c=clusters, cmap=c)
plt.title(f"Single linkage X2:")
plt.show()


# ----------------
# Average Linkage.
# ----------------

ac = AgglomerativeClustering(linkage='average', n_clusters=2)
clusters = ac.fit_predict(X1)

plt.scatter(d1['X1'], d1['X2'], c=clusters, cmap=c)
plt.title(f"Average linkage X1:")
plt.show()

ac = AgglomerativeClustering(linkage='average', n_clusters=2)
clusters = ac.fit_predict(X2)

plt.scatter(d2['X1'], d2['X2'], c=clusters, cmap=c)
plt.title(f"Average linkage X2:")
plt.show()




for linkage in ['single', 'complete', 'average']:
    
    # setting distance_threshold=0 ensures we compute the FULL TREE.
    ac = AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None)
    ac = ac.fit(X2)

    plot_dendrogram(ac, truncate_mode='level', p=4)
    plt.title(f"{linkage} linkage X2")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
"""


