"""
source : https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
"""

import numpy as np

def kmeans(data_set, k, max_iterations=100):
    # 1. Initialize centroids randomly by picking k points from the data
    num_samples, num_features = data_set.shape
    random_indices = np.random.choice(num_samples, k, replace=False)
    centroids = data_set[random_indices]
    
    iterations = 0
    old_centroids = None
    
    # Run the main k-means algorithm
    while not should_stop(old_centroids, centroids, iterations, max_iterations):
        old_centroids = centroids.copy()
        iterations += 1
        
        # 2. Assign labels based on closest centroid
        labels = get_labels(data_set, centroids)
        
        # 3. Calculate new centroids from the mean of assigned points
        centroids = get_centroids(data_set, labels, k)
        
    return centroids, labels

def should_stop(old_centroids, centroids, iterations, max_iterations):
    if iterations >= max_iterations: 
        return True
    if old_centroids is None: 
        return False
    # Stop if centroids don't change at all
    return np.array_equal(old_centroids, centroids)

def get_labels(data_set, centroids):
    # Calculate Euclidean distance from every point to every centroid
    # Using broadcasting: (num_samples, 1, num_features) - (1, k, num_features)
    distances = np.linalg.norm(data_set[:, np.newaxis] - centroids, axis=2)
    # Return the index of the minimum distance for each point
    return np.argmin(distances, axis=1)

def get_centroids(data_set, labels, k):
    num_features = data_set.shape[1]
    new_centroids = np.zeros((k, num_features))
    
    for i in range(k):
        # Get all points assigned to this cluster
        cluster_points = data_set[labels == i]
        
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            # Re-initialize randomly if a centroid becomes "empty"
            new_centroids[i] = data_set[np.random.choice(len(data_set))]
            
    return new_centroids