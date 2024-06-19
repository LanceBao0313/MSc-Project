import numpy as np

# Number of devices
num_devices = 100

# Generate random locations for devices (latitude and longitude)
np.random.seed(0)
device_locations = np.random.rand(num_devices, 2) * 100  # scaled to a 100x100 area

# Number of clusters
num_clusters = 5

# Randomly select initial cluster heads
initial_indices = np.random.choice(num_devices, num_clusters, replace=False)
cluster_heads = device_locations[initial_indices]


def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def assign_clusters(device_locations, cluster_heads):
    clusters = [[] for _ in range(len(cluster_heads))]
    for device in device_locations:
        #device = device + np.random.normal(0, 0.5, 2)  # Add noise to the device location
        distances = [calculate_distance(device, head) for head in cluster_heads]
        closest_cluster = np.argmin(distances)
        clusters[closest_cluster].append(device)
    return clusters

def update_cluster_heads(clusters):
    new_heads = []
    for cluster in clusters:
        if len(cluster) > 0:
            new_heads.append(np.mean(cluster, axis=0))
        else:
            new_heads.append(np.random.rand(1, 2) * 100)  # Assign a new random head if the cluster is empty
    return np.array(new_heads)

# Perform the clustering process
iterations = 30
for i in range(iterations):
    clusters = assign_clusters(device_locations, cluster_heads)
    new_cluster_heads = update_cluster_heads(clusters)
    if np.allclose(cluster_heads, new_cluster_heads):
        # Visualization
        #plt.scatter(device_locations[:, 0], device_locations[:, 1], c='white', label='Devices')
        break
    cluster_heads = new_cluster_heads

    # Visualization
    #plt.scatter(device_locations[:, 0], device_locations[:, 1], c='white', label='Devices')