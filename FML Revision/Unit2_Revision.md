# Unit 2: Regression and Generalization - Revision Notes

## 1. Bias-Variance Tradeoff

*   **Bias**: The error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
*   **Variance**: The error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).
*   **Tradeoff**:
    *   Increasing model complexity decreases bias but increases variance.
    *   Decreasing model complexity increases bias but decreases variance.
    *   The goal is to find a balance (optimal complexity) that minimizes the total error.

## 2. Overfitting and Underfitting

*   **Underfitting**: The model is too simple to capture the underlying structure of the data. It performs poorly on both training and test data.
*   **Overfitting**: The model is too complex and learns the training data too well, including its noise. It performs well on training data but poorly on new, unseen data (test data).

## 3. Statistical Decision Theory

*   A framework for making decisions in the presence of uncertainty.
*   It models decision-making as a process of choosing an action from a set of possible actions, where the outcome depends on an unknown state of nature.
*   **Key Components**:
    *   **Action Space (A)**: Set of all possible actions.
    *   **State of Nature (Θ)**: The true, unknown state of the world.
    *   **Loss Function L(θ, a)**: Quantifies the penalty for taking action 'a' when the true state is 'θ'.
    *   **Decision Rule (δ)**: A function that maps an observation 'x' to an action 'a'.
*   **Goal**: To find a decision rule that minimizes the expected loss (risk).

## 4. Regression vs. Classification

| Feature             | Regression                               | Classification                               |
| ------------------- | ---------------------------------------- | ---------------------------------------------- |
| **Output**          | Continuous value (e.g., price, temperature) | Discrete category (e.g., "cat", "dog", "spam") |
| **Goal**              | Predict a quantity.                      | Predict a class label.                         |
| **Example Algorithm** | Linear Regression                        | Logistic Regression, SVM, Decision Trees       |
| **Evaluation**        | MSE, R-squared                           | Accuracy, Precision, Recall, F1-Score          |

## 5. Linear Regression

*   **Goal**: To model the linear relationship between a dependent variable (y) and one or more independent variables (X).
*   **Equation**:  `y = β₀ + β₁x₁ + ... + βₙxₙ + ε`
*   **Cost Function (Mean Squared Error - MSE)**:
    $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$
*   **Gradient Descent**: An iterative optimization algorithm to find the values of θ that minimize the cost function. It repeatedly updates θ in the direction of the steepest descent of the cost function.
    *   **Update Rule**: `θ_j := θ_j - α * (∂J(θ) / ∂θ_j)` where α is the learning rate.

## 6. Logistic Regression

*   **Goal**: A classification algorithm used to predict the probability of a categorical dependent variable.
*   **Sigmoid Function**: Maps any real value into a value between 0 and 1.
    $$ g(z) = \frac{1}{1 + e^{-z}} $$
*   **Hypothesis**: $h_\theta(x) = g(\theta^T x)$
*   **Cost Function (Log Loss or Binary Cross-Entropy)**:
    $$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] $$
*   **Optimization**: Gradient Descent is used to find the parameters $\theta$ that minimize this cost function.

## 7. Normal Equation

*   An analytical approach to solving for the optimal parameters ($\theta$) in linear regression without iteration.
*   **Formula**: $\theta = (X^T X)^{-1} X^T y$
*   **Pros**: No need to choose a learning rate, no iterations.
*   **Cons**: Computationally expensive for a large number of features (inverting $X^T X$ is O(n³)).

## 8. Eigenvalues and Eigenvectors

*   For a square matrix A, an eigenvector **v** and its corresponding eigenvalue λ satisfy the equation: $Av = \lambda v$.
*   Eigenvectors represent the directions in which a linear transformation acts by stretching/compressing, and eigenvalues are the scalars by which this stretching/compression occurs.
*   They are fundamental to many dimensionality reduction techniques like PCA.
