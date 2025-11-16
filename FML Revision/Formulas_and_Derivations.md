# Machine Learning: Key Formulas and Derivations

This document provides a summary of important formulas and derivations from the course material.

---

## Unit 2: Regression and Generalization

### Linear Regression

*   **Hypothesis Function**:
    $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n = \theta^T x$

*   **Cost Function (Mean Squared Error - MSE)**:
    Measures the average squared difference between the estimated values and the actual value.
    $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$
    where $m$ is the number of training examples.

*   **Gradient Descent (Update Rule)**:
    An iterative optimization algorithm to find the minimum of the cost function.
    For each parameter $\theta_j$:
    $$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$
    $$ \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} $$
    where $\alpha$ is the learning rate.

*   **Normal Equation**:
    A non-iterative method to find the optimal $\theta$ that minimizes the cost function.
    $$ \theta = (X^T X)^{-1} X^T y $$
    where $X$ is the design matrix (with an intercept term) and $y$ is the vector of target values.

### Logistic Regression

*   **Hypothesis Function (Sigmoid/Logistic Function)**:
    Maps any real-valued number into a value between 0 and 1.
    $$ h_\theta(x) = g(z) = \frac{1}{1 + e^{-z}} \quad \text{where } z = \theta^T x $

*   **Cost Function (Log Loss or Binary Cross-Entropy)**:
    A convex function used for classification problems.
    $$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] $$
    where $y^{(i)}$ is the actual label (0 or 1).

---

## Unit 3: Dimensionality Reduction and Classification

### Principal Component Analysis (PCA)

*   **Goal**: Find a new set of orthogonal axes (principal components) that maximize the variance in the data.
*   **Process**:
    1.  **Standardize Data**: Center the data by subtracting the mean from each feature.
    2.  **Covariance Matrix**: Compute the covariance matrix $\Sigma$ of the standardized data.
    3.  **Eigen-decomposition**: Find the eigenvectors ($v_1, v_2, \dots, v_n$) and corresponding eigenvalues ($\lambda_1, \lambda_2, \dots, \lambda_n$) of $\Sigma$.
    4.  **Projection**: The principal components are the eigenvectors. The first principal component ($PC_1$) is the eigenvector with the largest eigenvalue, and so on. The transformed data is obtained by projecting the original data onto these eigenvectors.

### Linear Discriminant Analysis (LDA)

*   **Goal**: Find a feature subspace that maximizes the separability between classes.
*   **Process**:
    1.  **Compute Scatter Matrices**:
        *   **Within-class scatter matrix ($S_W$)**: $S_W = \sum_{i=1}^{c} \sum_{x \in D_i} (x - m_i)(x - m_i)^T$
        *   **Between-class scatter matrix ($S_B$)**: $S_B = \sum_{i=1}^{c} N_i (m_i - m)(m_i - m)^T$
        (where $c$ is the number of classes, $D_i$ is the set of samples in class $i$, $m_i$ is the mean of class $i$, $N_i$ is the number of samples in class $i$, and $m$ is the overall mean).
    2.  **Solve Eigenvalue Problem**: Solve the generalized eigenvalue problem for $S_W^{-1}S_B$. The eigenvectors are the linear discriminants.

### Support Vector Machine (SVM)

*   **Hyperplane**: $w^T x + b = 0$
*   **Margin**: The distance between the two support vectors, which is $\frac{2}{||w||}$.
*   **Optimization Problem (Hard Margin)**:
    Minimize $\frac{1}{2} ||w||^2$ subject to $y_i(w^T x_i + b) \ge 1$ for all $i$.
*   **Cost Function (Soft Margin)**:
    $$ J(w, b, \xi) = \frac{1}{2}||w||^2 + C \sum_{i=1}^{N} \xi_i $$
    Subject to $y_i(w^T x_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$.
    *   $C$ is the regularization parameter, balancing margin maximization and misclassification penalty.
    *   $\xi_i$ are slack variables allowing for some misclassification.

---

## Unit 4: Neural Networks and Decision Trees

### Backpropagation

*   **Chain Rule for Gradient Calculation**: The core of backpropagation. For a simple network $y = f(g(h(x)))$, the derivative of the loss $L$ with respect to an early weight $w$ is:
    $$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial w} $$
*   **Weight Update Rule (Gradient Descent)**:
    $$ w_{new} = w_{old} - \eta \frac{\partial L}{\partial w_{old}} $$
    where $\eta$ is the learning rate.

### Decision Tree Splitting Criteria

*   **Gini Impurity**: $Gini = 1 - \sum_{i=1}^{C} (p_i)^2$
*   **Entropy**: $H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$
*   **Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$

---

## Unit 5: Clustering and Ensemble Methods

### K-Means Clustering

*   **Objective Function (Within-Cluster Sum of Squares - WCSS)**:
    $$ J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 $$
    where $k$ is the number of clusters, $C_i$ is the $i$-th cluster, and $\mu_i$ is the centroid of cluster $C_i$.

### Gaussian Mixture Models (GMM)

*   **Probability Density Function (PDF)**:
    $$ P(x) = \sum_{k=1}^{K} \phi_k \cdot \mathcal{N}(x | \mu_k, \Sigma_k) $$
    where $\phi_k$ is the mixing coefficient, and $\mathcal{N}(x | \mu_k, \Sigma_k)$ is the Gaussian PDF for component $k$.

*   **Expectation-Maximization (EM) Algorithm**:
    *   **E-Step (Responsibility Calculation)**:
        $$ \gamma(z_{ik}) = \frac{\phi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \phi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)} $$
    *   **M-Step (Parameter Update)**:
        *   $\phi_k^{new} = \frac{1}{N} \sum_{i=1}^{N} \gamma(z_{ik})$
        *   $\mu_k^{new} = \frac{\sum_{i=1}^{N} \gamma(z_{ik}) x_i}{\sum_{i=1}^{N} \gamma(z_{ik})}$
        *   $\Sigma_k^{new} = \frac{\sum_{i=1}^{N} \gamma(z_{ik}) (x_i - \mu_k^{new}) (x_i - \mu_k^{new})^T}{\sum_{i=1}^{N} \gamma(z_{ik})}$

### Hypothesis Testing

*   **Z-statistic**: $Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$ (when population standard deviation $\sigma$ is known)
*   **t-statistic**: $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$ (when population standard deviation is unknown and estimated by sample standard deviation $s$)
