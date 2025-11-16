# Unit 3: Dimensionality Reduction & Classification - Revision Notes

## 1. Dimensionality Reduction

*   **Goal**: To reduce the number of input variables (features) in a dataset while preserving as much of the original information as possible.
*   **Why?**:
    *   **Curse of Dimensionality**: Performance of many algorithms degrades with a high number of features.
    *   **Computational Efficiency**: Fewer dimensions mean faster training and prediction.
    *   **Noise Reduction**: Can remove irrelevant or redundant features.
    *   **Visualization**: Allows for plotting data in 2D or 3D.

### Principal Component Analysis (PCA)

*   **Type**: Unsupervised, linear dimensionality reduction technique.
*   **Goal**: To find a new set of orthogonal axes (principal components) that capture the maximum variance in the data.
*   **How it works**:
    1.  **Standardize Data**: Scale the data to have zero mean and unit variance.
    2.  **Covariance Matrix**: Compute the covariance matrix of the standardized data.
    3.  **Eigen-decomposition**: Calculate the eigenvectors and eigenvalues of the covariance matrix.
    4.  **Select Principal Components**: The eigenvectors with the highest eigenvalues are the principal components. They represent the directions of maximum variance.
    5.  **Transform Data**: Project the original data onto the selected principal components to get a lower-dimensional representation.

### Linear Discriminant Analysis (LDA)

*   **Type**: Supervised dimensionality reduction and classification technique.
*   **Goal**: To find a feature subspace that maximizes the separability between different classes.
*   **How it works**:
    1.  **Compute Scatter Matrices**:
        *   **Within-class scatter matrix (S_W)**: Measures the spread of data within each class.
        *   **Between-class scatter matrix (S_B)**: Measures the separation between the means of different classes.
    2.  **Solve Eigenvalue Problem**: Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$.
    3.  **Select Linear Discriminants**: The eigenvectors corresponding to the largest eigenvalues are the linear discriminants.
    4.  **Transform Data**: Project the data onto the new subspace defined by these discriminants.
*   **PCA vs. LDA**:
    *   **PCA** is unsupervised and finds directions of maximum variance.
    *   **LDA** is supervised and finds directions that maximize class separability.

## 2. Subset Selection

*   **Goal**: To find the best subset of the original features to use in a model.
*   **Types**:
    *   **Best Subset Selection**: Tries all possible combinations of features. Computationally very expensive.
    *   **Forward Stepwise Selection**: Starts with no features and adds them one by one, choosing the one that improves the model the most at each step.
    *   **Backward Stepwise Selection**: Starts with all features and removes them one by one, eliminating the one that has the least impact on performance.

## 3. Shrinkage (Regularization) Methods

*   **Goal**: To reduce model complexity and prevent overfitting by shrinking the coefficient estimates towards zero.
*   **Types**:
    *   **Ridge Regression (L2 Regularization)**: Adds a penalty equal to the square of the magnitude of coefficients. It shrinks coefficients but doesn't set them to zero.
    *   **Lasso Regression (L1 Regularization)**: Adds a penalty equal to the absolute value of the magnitude of coefficients. It can shrink some coefficients to exactly zero, effectively performing feature selection.

## 4. Support Vector Machines (SVM)

*   **Concept**: A powerful supervised learning algorithm for classification and regression.
*   **Goal**: To find an optimal **hyperplane** that separates data points of different classes with the maximum possible **margin** (the distance between the hyperplane and the nearest data points of any class).
*   **Support Vectors**: The data points that lie closest to the hyperplane and define its position.
*   **Kernel Trick**: A key feature of SVMs that allows them to handle non-linearly separable data. It maps the data into a higher-dimensional space where a linear separator can be found. Common kernels include Linear, Polynomial, and Radial Basis Function (RBF).
*   **Cost Function**: The SVM's objective is to minimize the norm of the weight vector (to maximize the margin) while penalizing misclassifications. This is often expressed using a hinge loss function and a regularization term.

## 5. Linear vs. Logistic Regression

| Feature             | Linear Regression                               | Logistic Regression                               |
| ------------------- | ----------------------------------------------- | ------------------------------------------------- |
| **Output**          | Continuous (e.g., price, temperature)         | Probability (0 to 1), then a discrete class (0 or 1) |
| **Purpose**         | Prediction of a value                           | Classification                                    |
| **Equation**        | Linear equation ($y = mx + c$)                  | Sigmoid function of a linear equation             |
| **Cost Function**   | Mean Squared Error (MSE)                        | Log Loss (Binary Cross-Entropy)                   |
| **Decision Boundary** | The regression line itself is the prediction. | A linear decision boundary separates classes.      |
