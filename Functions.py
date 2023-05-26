import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('data/diabetes.csv')
# reduce samples to 600
data = data.sample(500)
X = data.drop('Outcome', axis=1).to_numpy()
y = data['Outcome'].to_numpy()


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


# Inner function for sammon algorithm
def update_layout(X, y, alpha, c, min_threshold):
    partial1, partial2 = np.zeros(2), np.zeros(2)
    for j in range(X.shape[0]):
        for k in range(X.shape[0]):
            if k != j:
                x_diff, y_diff = np.subtract(X[j], X[k]), np.subtract(y[j], y[k])
                delta_x, delta_y = np.linalg.norm(x_diff), np.linalg.norm(y_diff)
                divergence, denominator = np.subtract(delta_x, delta_y), np.multiply(delta_x, delta_y)
                # limiting the denominator to a minimum value
                denominator = np.maximum(denominator, min_threshold)
                # calculating the partial derivatives
                partial1 += (divergence / denominator) * y_diff
                partial2 += (1 / denominator) * (
                        divergence - (((y_diff ** 2) / delta_y) * (1 + (divergence / delta_y))))

        y_update = (((-2 / c) * partial1) / np.abs(((-2 / c) * partial2)))
        y[j] -= alpha * y_update
    return y

# 0005 not bad
def sammon(X, iterations, error, alpha):
    min_threshold = 1e-6
    # 1. Start with a random two-dimensional layout Y of points (Y is a n × 2 matrix).
    y = np.random.normal(0, 1, size=(X.shape[0], 2))
    x_dist = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    c = np.sum(x_dist) / 2
    for i in range(iterations):
        y_dist = np.linalg.norm(y[:, np.newaxis] - y, axis=2)
        # 2. Compute the stress E of Y after every 5 iterations
        if i % 5 == 0:
            E = (np.sum((x_dist - y_dist) ** 2) / np.sum(x_dist)) / 2
        # 3. If E < e, or if the maximum number of iterations iter has been reached, stop.
            if E < error:
                return y
        # 4. For each yi of Y , find the next vector yi(t + 1) based on the current yi(t)
        y = update_layout(X, y, alpha, c, min_threshold)
    return y


def generate_dr_results(X_values, iterations, error, alpha):
    results = {}
    for i, X in enumerate(X_values):
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)
        pca_result = pca.fit_transform(X)
        tsne_result = tsne.fit_transform(X)
        sammon_result = sammon(X, iterations, error, alpha)

        results[f'pca_X{i}'] = pca_result
        results[f'tsne_X{i}'] = tsne_result
        results[f'sammon_X{i}'] = sammon_result
    return results
