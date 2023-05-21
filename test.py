import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Function which returns the largest cluster to divide
def largest_cluster(indices):
    # Basically finds the most common cluster in the array and returns its corresponding number
    counts = np.bincount(indices)
    return np.argmax(counts)

# Bisecting k-Means function
def bkmeans(X,k,iter):
    minSSE = 0
    countClusters = 0
    largestCluster = 0
    finalCluster = np.zeros(X.shape[0], dtype=int)
    clsLabels = np.zeros(X.shape[0])

    # Run this until we have k clusters
    while (countClusters < k-1):

        # Divide the largest cluster into two smaller sub-clusters iter times
        for i in range(iter):
            kmeans = KMeans(n_clusters=2).fit(X[clsLabels == largestCluster])

            # Choose the best solution according to SSE
            tempSSE = kmeans.inertia_
            if (i == 0):
                minSSE = tempSSE
                subcluster_indices = kmeans.labels_

            if (minSSE > tempSSE):
                minSSE = tempSSE
                subcluster_indices = kmeans.labels_

        # Assign new colors to the sub-clusters because it cant be 0 and 1 since it already exists
        subcluster_indices = np.where(subcluster_indices == 1, countClusters + 1, subcluster_indices)
        subcluster_indices = np.where(subcluster_indices == 0, largestCluster, subcluster_indices)

        # Merge the two new sub-clusters into the main array
        index = 0
        for n, i in enumerate(finalCluster):
            if i == largestCluster:
                number = subcluster_indices[index]
                finalCluster[n] = number
                index += 1

        # Prepare values for next round
        countClusters += 1
        largestCluster = largest_cluster(finalCluster)
        clsLabels = finalCluster

    return finalCluster

# Creating some random data
my_data = make_blobs(n_samples=100, n_features=2)
X = my_data[0]
result = bkmeans(X,5,100)

print('Output vector:')
print(result)

# Plotting the clusters
plt.scatter(X[:, 0],X[:, 1], c=result, cmap='rainbow')
plt.show()