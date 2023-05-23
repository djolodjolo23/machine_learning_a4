import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def bkmeans(x_data, num_of_clusters, iterations):
    global cluster1, sse2, sse1, cluster2, leftover_indices
    normalized_x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    final_indices = np.zeros((x_data.shape[0]))
    current_x_data = normalized_x_data
    min_sse = 0
    for i in range(1, num_of_clusters):
        for j in range(iterations):
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(current_x_data)
            current_sse = kmeans.inertia_
            if current_sse < min_sse or min_sse == 0:
                min_sse = current_sse
                cluster1, cluster2 = current_x_data[kmeans.labels_ == 0], current_x_data[kmeans.labels_ == 1]
                sse1, sse2 = np.sum((cluster1 - np.mean(cluster1)) ** 2), np.sum((cluster2 - np.mean(cluster2)) ** 2)
                leftover_indices = np.copy(final_indices[final_indices == 0])
                if sse1 > sse2:
                    leftover_indices[kmeans.labels_ == 1] = i
                else:
                    leftover_indices[kmeans.labels_ == 0] = i
        if sse1 > sse2:
            current_x_data = cluster1
        else:
            current_x_data = cluster2
        final_indices[final_indices == 0] = leftover_indices
    final_indices[final_indices == 0] = num_of_clusters
    return final_indices


def sammon(x_data, iterations, error_threshold, learning_rate):
    Y = np.random.normal(0, 1, size=(x_data.shape[0], 2))
    for i in range(iterations):
        D = cdist(x_data, x_data)  # original space
        y_dist = cdist(Y, Y)  # target space
        # compute the stress E of Y.
        E_stress = (np.sum((D - y_dist) ** 2) / np.sum(D)) / 2  # stress E on y_dist
        if E_stress < error_threshold or iterations - 1 == i:
            return Y
        delta = D - y_dist
        scaling_factor = D * (y_dist + 1e-10)

        # Calculate the gradient for each point in the layout
        gradient = np.zeros((x_data.shape[0], 2))
        for j in range(x_data.shape[0]):
            for k in range(x_data.shape[0]):
                if j != k:
                    diff = Y[j] - Y[k]
                    distance = np.linalg.norm(diff)
                    factor = (delta[j, k] - distance) / (distance * (delta[j, k] + 1e-10))
                    gradient[j] += factor * diff
        Y -= learning_rate / np.sqrt(i + 1) * gradient
    return Y


def sammon_new(x_data, iterations, error_threshold, learning_rate):
    pass



