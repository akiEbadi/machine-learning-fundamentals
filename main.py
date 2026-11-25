import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Sample traffic events
# Each row represents one event.
# Features: speed, speed variation, direction change,
# duration, distance, and acceleration.

events = [
    [12, 1.2, 0.1, 8, 35, 0.2],
    [14, 1.5, 0.2, 9, 40, 0.3],
    [13, 1.1, 0.1, 7, 37, 0.2],
    [15, 1.7, 0.2, 8, 42, 0.4],
    [11, 1.3, 0.1, 9, 33, 0.2],

    [7, 2.5, 1.8, 6, 20, 1.5],
    [8, 2.2, 1.6, 7, 23, 1.3],
    [6, 2.8, 2.0, 5, 18, 1.8],
    [9, 2.4, 1.7, 6, 25, 1.4],
    [7, 2.6, 1.9, 7, 21, 1.6],

    [1, 0.3, 0.1, 12, 2, 0.1],
    [0.5, 0.2, 0.1, 15, 1, 0.1],
    [2, 0.5, 0.2, 10, 4, 0.2],
    [1.5, 0.4, 0.1, 13, 3, 0.1],
    [0.8, 0.3, 0.1, 14, 2, 0.1],

    [10, 3.5, 0.3, 5, 28, 2.5],
    [12, 4.0, 0.2, 6, 34, 3.0],
    [9, 3.2, 0.4, 5, 25, 2.2],
    [11, 3.8, 0.3, 7, 30, 2.8],
    [13, 4.2, 0.2, 6, 36, 3.2]
]

feature_names = [
    "Speed",
    "Speed variation",
    "Direction change",
    "Duration",
    "Distance",
    "Acceleration"
]

X = np.array(events, dtype=float)


def kmeans(X, k=4, max_iterations=100):
    # Pick initial centers from the data
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), k, replace=False)
    centers = X[indices].copy()

    for _ in range(max_iterations):
        # Find the distance from each point to every center
        distances = np.zeros((len(X), k))

        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centers[i], axis=1)

        # Assign points to their nearest center
        labels = np.argmin(distances, axis=1)

        # Move each center to the mean of its assigned points
        new_centers = centers.copy()

        for i in range(k):
            cluster_points = X[labels == i]

            if len(cluster_points) > 0:
                new_centers[i] = cluster_points.mean(axis=0)

        # Stop when the centers have settled
        if np.allclose(centers, new_centers):
            centers = new_centers
            break

        centers = new_centers

    # Reassign points using the final centers
    distances = np.zeros((len(X), k))
    for i in range(k):
        distances[:, i] = np.linalg.norm(X - centers[i], axis=1)
    labels = np.argmin(distances, axis=1)

    return labels, centers


# Scale features so that larger numeric ranges do not dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

labels, centers = kmeans(X_scaled, k=4)

print("Traffic Event Clustering")
print("---------------------")
for cluster in range(4):
    count = np.sum(labels == cluster)
    print(f"Cluster {cluster + 1}: {count} events")


# PCA is used only to display the data in two dimensions
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
centers_2d = pca.transform(centers)

plt.figure(figsize=(8, 6))

for cluster in range(4):
    points = X_2d[labels == cluster]
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster + 1}")

plt.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    marker="X",
    color="black",
    s=180,
    label="Cluster centers"
)

plt.title("Traffic Events - K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("traffic_clusters.png", dpi=300)
plt.show()


# Simple checks
assert len(labels) == len(X)
assert centers.shape == (4, X.shape[1])

print("\nAll checks passed.")
