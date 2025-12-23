import numpy as np
from sklearn.cluster import KMeans

hours = np.array([[9],[10],[11],[3],[2],[1],[32]])
lables = KMeans(n_clusters=3, n_init=10).fit(hours).labels_
# print(list(zip(hours.ravel(), lables)))

print([(int(h), int(l)) for h, l in zip(hours.ravel(), lables)])
