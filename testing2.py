import numpy as np
from sklearn import datasets
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

iris = datasets.load_iris()
#X = iris.data
#y = iris.target

def sammon(X, iter, error, alpha):

    # Number of samples
    n_samples = len(X)

    # 1. Random two-dimensional layout Y
    y = np.random.normal(0, 1, size=(X.shape[0], 2))
    distX = pdist(X, 'euclidean')
    sum_pairwise_dissimilarities = np.sum(distX)
    scaling_factor = -2 / sum_pairwise_dissimilarities
    # Loop iter times
    for x in range(iter):

        # Calculates all the distances of the output space

        distY = pdist(y, 'euclidean')

        # 2. Compute the stress E of Y
        E = (np.sum((distX - distY) ** 2) / np.sum(distX)) / 2  # stress E on y_dist

        # 3. If E < e --> stop
        if E < error or iter - 1 == x:
            return y

        # 4. For each yi of Y, find the next vector yi(t+1)

        partial1 = partial2 = np.zeros(2)
        #y_next = np.zeros((n_samples, 2))

        for i in range(n_samples):
            for j in range(n_samples):
                if (j != i):

                        # Differences needed further
                    X_diff, y_diff = np.subtract(X[i], X[j]), np.subtract(y[i], y[j])
                    delta_x, delta_y = np.linalg.norm(X_diff), np.linalg.norm(y_diff)
                    divergence = delta_x - delta_y
                    denominator = delta_y * delta_x
                    # Limits how small the denominator can be
                    min_threshold = 0.00001
                    denominator = np.maximum(denominator, min_threshold)

                        # Calculates the partial equations
                    partial1 += (divergence / denominator) * y_diff
                    partial2 += (1 / denominator) * (divergence - (((y_diff ** 2) / delta_y) * (1 + (divergence / delta_y))))

            deltai_t = (((-2 / sum_pairwise_dissimilarities) * partial1) / np.abs(((-2 / sum_pairwise_dissimilarities) * partial2)))
            #y_next[i] = y[i] - alpha * deltai_t
            y[i] -= alpha * deltai_t

        #y = np.copy(y_next)
    return y

X = make_blobs(n_samples=50, n_features=2)
y = X[1]
test = sammon(X[0], 100, 0.0, 0.3)

plt.figure(1)
plt.scatter(test[:, 0],test[:, 1], c=y, cmap='rainbow')
plt.show()


