# Unit 4: Neural Networks and Decision Trees - Revision Notes

## 1. Artificial Neural Networks (ANNs)

*   **Concept**: Computational models inspired by the structure and function of biological neural networks. They consist of interconnected nodes (neurons) organized in layers.
*   **Structure**:
    *   **Input Layer**: Receives the initial data.
    *   **Hidden Layers**: One or more layers between input and output that perform computations.
    *   **Output Layer**: Produces the final result.
*   **Key Components**:
    *   **Neurons (Nodes)**: Basic processing units.
    *   **Weights**: Parameters that determine the strength of the connection between neurons.
    *   **Bias**: An additional parameter to adjust the output, similar to an intercept.
    *   **Activation Function**: A function that determines the output of a neuron.

### 2. Backpropagation

*   **Purpose**: The primary algorithm for training ANNs. It calculates the gradient of the loss function with respect to the network's weights, allowing for efficient weight updates.
*   **Process**:
    1.  **Forward Pass**: Input data is fed through the network to generate an output.
    2.  **Calculate Loss**: The difference between the predicted output and the actual target is calculated using a loss function (e.g., Mean Squared Error, Cross-Entropy).
    3.  **Backward Pass**: The error is propagated backward from the output layer to the input layer. The chain rule of calculus is used to calculate the gradient of the loss with respect to each weight and bias.
    4.  **Update Weights**: The weights and biases are updated in the direction that minimizes the loss, using an optimization algorithm like Gradient Descent.

### 3. Perceptron

*   **Concept**: The simplest form of a neural network, consisting of a single neuron.
*   **Function**: It takes binary inputs, calculates a weighted sum, and applies a step function to produce a binary output.
*   **Limitation**: Can only solve linearly separable problems.

### 4. Activation Functions

*   **Purpose**: To introduce non-linearity into the network, allowing it to learn complex patterns.
*   **Common Types**:
    *   **Sigmoid**: S-shaped curve, outputs between 0 and 1. Prone to vanishing gradients.
    *   **Tanh (Hyperbolic Tangent)**: S-shaped curve, outputs between -1 and 1. Zero-centered, but still has vanishing gradient issues.
    *   **ReLU (Rectified Linear Unit)**: Outputs the input if positive, otherwise 0. Computationally efficient and widely used, but can suffer from the "dying ReLU" problem.
    *   **Leaky ReLU**: A variant of ReLU that allows a small, non-zero gradient for negative inputs.
    *   **Softmax**: Used in the output layer for multi-class classification. Converts a vector of scores into a probability distribution.

### 5. Optimization Functions (Optimizers)

*   **Purpose**: Algorithms used to update the weights and biases of the network to minimize the loss function.
*   **Common Types**:
    *   **Stochastic Gradient Descent (SGD)**: Updates weights based on the error of a single training example or a small batch.
    *   **Adam (Adaptive Moment Estimation)**: An adaptive learning rate method that computes individual learning rates for different parameters. It's a popular and effective choice for many problems.
    *   **RMSprop**: Another adaptive learning rate method that works well in practice.

## 6. Decision Trees

*   **Concept**: A tree-like model where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome (a class label).
*   **How it works**: It splits the data into subsets based on the values of input features, creating a tree structure.
*   **Splitting Criteria**: The process of selecting the best feature to split on at each node. The goal is to create the most homogeneous sub-nodes.
    *   **Gini Impurity**: Measures the probability of misclassifying a randomly chosen element. A lower Gini index means a purer node.
    *   **Information Gain (based on Entropy)**: Measures the reduction in uncertainty (entropy) after a dataset is split on an attribute. The split that results in the highest information gain is chosen.
*   **Advantages**: Easy to understand and interpret, can handle both numerical and categorical data.
*   **Disadvantages**: Prone to overfitting, can be unstable.

## 7. Ensemble Methods

*   **Concept**: Combine multiple machine learning models to produce a more robust and accurate model.
*   **Types**:
    *   **Bagging (e.g., Random Forest)**: Trains multiple models independently on different subsets of the data and averages their predictions. Reduces variance.
    *   **Boosting (e.g., AdaBoost, Gradient Boosting)**: Trains models sequentially, where each new model corrects the errors of the previous one. Reduces bias.
    *   **Stacking**: Uses a meta-model to learn how to best combine the predictions of multiple base models.
*   **Decision Trees vs. Ensemble Methods**:
    *   A single decision tree is a single model, while an ensemble method combines multiple models.
    *   Ensemble methods are generally more accurate and less prone to overfitting than a single decision tree.
    *   Decision trees are more interpretable than ensemble methods.

## 8. Bayesian Learning

*   **Concept**: A probabilistic approach to machine learning based on Bayes' Theorem. It updates the probability of a hypothesis based on new evidence.
*   **Bayes' Theorem**: $P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}$
*   **Impact**:
    *   Provides a way to quantify uncertainty in predictions.
    *   Allows for the incorporation of prior knowledge.
    *   Forms the basis for algorithms like Naive Bayes and Bayesian Networks.

## 9. Parameter Estimation

*   **Concept**: The process of using sample data to estimate the unknown parameters of a statistical model.
*   **Maximum Likelihood Estimation (MLE)**: A method for estimating the parameters of a statistical model by finding the parameter values that maximize the likelihood of making the observations given the parameters.
    *   **Example**: For a series of coin flips, the MLE for the probability of heads is the number of heads divided by the total number of flips.
*   **Bayesian Parameter Estimation**: Treats parameters as random variables and estimates their posterior distribution based on prior beliefs and observed data.
    *   **Advantages over MLE**: Incorporates prior knowledge, provides a full probability distribution for parameters (quantifying uncertainty), and can be more robust with small datasets.
