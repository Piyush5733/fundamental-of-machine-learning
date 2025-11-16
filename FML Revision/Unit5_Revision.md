# Unit 5: Clustering and Ensemble Methods - Revision Notes

## 1. Clustering

*   **Definition**: An unsupervised learning task that involves grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups.
*   **Goal**: To discover inherent groupings in the data without any predefined labels.

### Types of Clustering Algorithms

1.  **Partitioning Methods**:
    *   Divides the dataset into a pre-specified number of non-overlapping clusters.
    *   **K-Means**: A popular centroid-based algorithm. It iteratively assigns data points to the nearest cluster centroid and then recalculates the centroids as the mean of the assigned points. It aims to minimize the within-cluster sum of squares (WCSS).

2.  **Hierarchical Methods**:
    *   Creates a tree-like hierarchy of clusters (a dendrogram).
    *   **Agglomerative (Bottom-up)**: Starts with each data point as its own cluster and merges the closest pairs of clusters until only one cluster remains.
    *   **Divisive (Top-down)**: Starts with all data points in one cluster and recursively splits them into smaller clusters.

3.  **Density-Based Methods**:
    *   Defines clusters as dense regions of data points separated by sparser regions.
    *   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups together points that are closely packed, marking as outliers points that lie alone in low-density regions. It can find arbitrarily shaped clusters.

4.  **Model-Based Methods**:
    *   Assumes that the data is a mixture of underlying probability distributions.
    *   **Gaussian Mixture Models (GMMs)**: A probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions. It uses the Expectation-Maximization (EM) algorithm to find the parameters of these distributions. GMMs perform "soft" clustering, assigning a probability of membership to each cluster for every data point.

5.  **Spectral Clustering**:
    *   A technique that uses the eigenvalues (spectrum) of a similarity matrix of the data to perform dimensionality reduction before clustering in fewer dimensions.
    *   **Advantages**: Very effective at finding non-convex (non-globular) cluster shapes and can be more robust to noise.

## 2. Ensemble Methods

*   **Concept**: A machine learning technique where multiple models (often called "weak learners" or "base models") are trained to solve the same problem and their predictions are combined to get a better overall result.
*   **Goal**: To improve the accuracy, robustness, and generalization of a model by reducing bias and/or variance.

### Key Ensemble Techniques

1.  **Bagging (Bootstrap Aggregating)**:
    *   **How it works**: Creates multiple subsets of the original dataset by sampling with replacement (bootstrapping). A base model is trained independently on each subset. The final prediction is made by averaging (for regression) or taking a majority vote (for classification) of all the individual model predictions.
    *   **Effect**: Primarily reduces **variance** and helps to prevent overfitting.
    *   **Example**: **Random Forest**, which is an ensemble of decision trees.

2.  **Boosting**:
    *   **How it works**: Builds a sequence of models, where each new model attempts to correct the errors of its predecessor. It gives more weight to the data points that were misclassified by previous models.
    *   **Effect**: Primarily reduces **bias** and builds a strong classifier from a number of weak ones.
    *   **Examples**: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM.

3.  **Stacking (Stacked Generalization)**:
    *   **How it works**: Involves training several different models (e.g., a logistic regression, a decision tree, and a support vector machine) on the same data. A "meta-model" is then trained on the outputs (predictions) of these base models to make the final prediction.
    *   **Effect**: Aims to combine the strengths of different types of models to achieve the best possible performance.

## 3. Hypothesis Testing

*   **Definition**: A statistical method used to make decisions or draw conclusions about a population based on a sample of data.
*   **Core Components**:
    *   **Null Hypothesis ($H_0$)**: The default assumption, often stating no effect or no difference.
    *   **Alternative Hypothesis ($H_a$ or $H_1$)**: The claim to be tested, which contradicts the null hypothesis.
    *   **Significance Level ($\alpha$)**: The probability of rejecting the null hypothesis when it is true (Type I error). Commonly set to 0.05.
    *   **P-value**: The probability of observing the collected data, or something more extreme, if the null hypothesis were true. A small p-value (e.g., < 0.05) provides evidence against the null hypothesis.
*   **Types of Errors**:
    *   **Type I Error (False Positive)**: Rejecting a true null hypothesis.
    *   **Type II Error (False Negative)**: Failing to reject a false null hypothesis.
*   **Process**:
    1.  State the null and alternative hypotheses.
    2.  Choose a significance level ($\alpha$).
    3.  Calculate a test statistic from the sample data.
    4.  Determine the p-value.
    5.  Compare the p-value to $\alpha$ and make a decision: reject or fail to reject the null hypothesis.
    6.  Draw a conclusion in the context of the problem.
