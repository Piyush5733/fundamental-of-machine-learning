# Short Notes on Clustering and Hypothesis Testing

## a) Clustering

**Clustering** is an unsupervised machine learning technique used to group a set of unlabeled data points into subsets (clusters) based on their inherent similarities. The goal is to organize data such that points within the same cluster are more similar to each other than to those in other clusters. It helps in discovering hidden patterns, structures, or relationships within data without prior knowledge of class labels.

**Key Aspects:**
*   **Unsupervised Learning**: No pre-defined labels are used for training.
*   **Similarity-Based Grouping**: Data points are grouped based on a measure of similarity or distance.
*   **Types**: Common methods include:
    *   **Partitioning Methods** (e.g., K-Means): Divide data into a fixed number of clusters, with each point belonging to one cluster.
    *   **Hierarchical Methods**: Build a tree-like hierarchy of clusters (dendrogram).
    *   **Density-Based Methods** (e.g., DBSCAN): Identify clusters as dense regions separated by sparser areas, capable of finding arbitrary shapes and handling noise.
    *   **Distribution Model-Based Methods** (e.g., GMMs): Assume data points are generated from a mixture of probability distributions, providing soft cluster assignments.
*   **Applications**: Customer segmentation, anomaly detection, image segmentation, document analysis.

## b) Hypothesis Testing

**Hypothesis testing** is a statistical method used to make inferences about a population based on sample data. It's a formal procedure to assess the validity of a claim or assumption (a hypothesis) about a population parameter, determining whether observed results are statistically significant or could have occurred by chance.

**Key Steps:**
1.  **Formulate Hypotheses**: Define a **Null Hypothesis ($H_0$)** (no effect/difference) and an **Alternative Hypothesis ($H_1$)** (there is an effect/difference).
2.  **Set Significance Level ($\alpha$)**: A threshold (e.g., 0.05) for rejecting $H_0$.
3.  **Collect Data and Calculate Test Statistic**: Compute a value from sample data that measures deviation from $H_0$.
4.  **Determine P-value**: The probability of observing data as extreme as, or more extreme than, the sample data, assuming $H_0$ is true.
5.  **Make a Decision**:
    *   If P-value $\le \alpha$: Reject $H_0$ (evidence supports $H_1$).
    *   If P-value $> \alpha$: Fail to reject $H_0$ (insufficient evidence against $H_0$).
*   **Types of Errors**:
    *   **Type I Error ($\alpha$)**: Rejecting a true $H_0$ (false positive).
    *   **Type II Error ($\beta$)**: Failing to reject a false $H_0$ (false negative).
*   **Importance**: Crucial for validating claims, guiding data-driven decisions, and advancing scientific knowledge.