# Centroid-Based Clustering Algorithms: K-Means

**Centroid-based clustering algorithms** are a category of partitioning clustering methods that organize data into non-hierarchical clusters. The core idea is to represent each cluster by a central vector, or **centroid**, which is typically the mean position of all data points belonging to that cluster. Data points are then assigned to the cluster whose centroid is closest to them.

The most prominent and widely used centroid-based clustering algorithm is **K-Means**.

## K-Means Clustering Algorithm

The K-Means algorithm is an unsupervised learning algorithm used to partition $n$ observations into $k$ clusters, where each observation belongs to the cluster with the nearest mean (centroid). The number of clusters, $k$, is a pre-defined parameter.

### How K-Means Works (Algorithm Steps):

1.  **Initialization (Choose K and Initial Centroids)**:
    *   **Specify K**: The user first decides on the number of clusters, $K$, that the algorithm should form.
    *   **Initialize Centroids**: $K$ initial centroids are chosen. This step is crucial and can significantly impact the final clustering result. Common methods include:
        *   **Randomly selecting K data points** from the dataset as initial centroids.
        *   **K-Means++**: A smarter initialization technique that selects initial centroids that are far apart from each other, leading to more stable and accurate results.

2.  **Assignment Step (Assign Data Points to Clusters)**:
    *   Each data point in the dataset is assigned to the closest centroid. The "closest" is typically determined using a distance metric, most commonly Euclidean distance.
    *   For each data point $x_i$, calculate its distance to every centroid $c_j$.
    *   Assign $x_i$ to the cluster $j$ for which $dist(x_i, c_j)$ is minimal.

3.  **Update Step (Recalculate Centroids)**:
    *   After all data points have been assigned to clusters, the centroids of the clusters are recalculated.
    *   The new centroid for each cluster is the mean (average) of all data points currently assigned to that cluster.
    *   $$ c_j = \frac{1}{|C_j|} \sum_{x \in C_j} x $$
        Where $c_j$ is the new centroid for cluster $j$, and $C_j$ is the set of data points assigned to cluster $j$.

4.  **Iteration and Convergence**:
    *   Steps 2 and 3 are repeated iteratively.
    *   The algorithm converges when the assignments of data points to clusters no longer change, or when the centroids no longer move significantly, or when a maximum number of iterations is reached.

### Objective Function (Within-Cluster Sum of Squares - WCSS):

The K-Means algorithm aims to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as inertia. This measures the sum of squared distances between each data point and its assigned centroid.
$$ WCSS = \sum_{j=1}^{K} \sum_{x \in C_j} ||x - c_j||^2 $$
The algorithm tries to find cluster assignments and centroids that minimize this value.

### Advantages of K-Means:

*   **Simplicity**: Relatively easy to understand and implement.
*   **Efficiency**: Computationally efficient for large datasets, especially when $K$ is small.
*   **Scalability**: Can be scaled to large datasets.
*   **Versatility**: Widely applicable across various domains.

### Limitations of K-Means:

*   **Requires Pre-defined K**: The number of clusters ($K$) must be specified in advance, which is often unknown in real-world scenarios. Techniques like the Elbow Method or Silhouette Score can help determine an optimal $K$.
*   **Sensitivity to Initial Centroids**: The final clustering result can be sensitive to the initial random placement of centroids. Different initializations can lead to different local optima. K-Means++ helps mitigate this.
*   **Sensitivity to Outliers**: Outliers can significantly distort the position of centroids, leading to inaccurate cluster boundaries.
*   **Assumes Spherical Clusters**: K-Means works best when clusters are spherical, of similar size, and have similar densities. It struggles with irregularly shaped clusters or clusters with varying densities.
*   **Hard Assignment**: Each data point is assigned to exactly one cluster (hard clustering), which might not be ideal for data points that lie ambiguously between clusters.

## Conclusion

K-Means is a powerful and popular centroid-based clustering algorithm due to its simplicity and efficiency. While it has limitations, particularly regarding its assumptions about cluster shape and the need to specify $K$, its effectiveness in many practical applications makes it a cornerstone of unsupervised learning. Improvements like K-Means++ address some of its weaknesses, making it even more robust.