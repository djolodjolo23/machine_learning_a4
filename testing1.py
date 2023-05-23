import numpy as np
from sklearn import datasets
import numpy as np

from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data
y = iris.target



def bkmeans_new(X_data, num_of_clusters, iterations):
    global cluster1, sse2, sse1, cluster2
    indices = np.zeros((X_data.shape[0]))
    current_x_data = X_data
    min_sse = 0
    for i in range(1, num_of_clusters):
        for j in range(iterations):
            kmeans = KMeans(n_clusters=2, random_state=0).fit(current_x_data)
            current_sse = kmeans.inertia_
            if current_sse < min_sse or min_sse == 0:
                min_sse = current_sse
                cluster1, cluster2 = current_x_data[kmeans.labels_ == 0], current_x_data[kmeans.labels_ == 1]
                sse1, sse2 = np.sum((cluster1 - np.mean(cluster1))**2), np.sum((cluster2 - np.mean(cluster2))**2)
                leftover_indices = np.copy(indices[indices == 0])
                if sse1 > sse2:
                    leftover_indices[kmeans.labels_ == 1] = i
                else:
                    leftover_indices[kmeans.labels_ == 0] = i
        if sse1 > sse2:
            current_x_data = cluster1
        else:
            current_x_data = cluster2
        indices[indices == 0] = leftover_indices
    indices[indices == 0] = num_of_clusters
    return indices

result = bkmeans_new(X, 4, 20)

# plot the result

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=result, s=50, cmap='viridis')
plt.show()