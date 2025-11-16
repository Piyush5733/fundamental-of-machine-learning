# Unit 1: Introduction to Machine Learning - Revision Notes

## 1. What is Machine Learning?
Machine Learning (ML) is a subfield of artificial intelligence (AI) that focuses on building systems that can learn from data, identify patterns, and make decisions with minimal human intervention. Instead of being explicitly programmed, these systems use algorithms and statistical models to analyze and draw inferences from patterns in data.

**Key Concepts:**
- **Learning from Data:** ML algorithms are trained on large datasets to recognize patterns.
- **Prediction/Decision Making:** Once trained, the model can make predictions or decisions on new, unseen data.
- **Core Components:**
    - **Data:** The fuel for any ML model.
    - **Model:** The mathematical representation of the real-world process.
    - **Objective Function:** A function that measures the model's performance (e.g., error or loss).
    - **Learning Algorithm:** The process used to optimize the model's parameters to minimize the error.

## 2. Types of Machine Learning

There are three main categories of machine learning:

### a) Supervised Learning
- **Concept:** The algorithm learns from a labeled dataset, meaning each data point is tagged with a correct output or target. The goal is to learn a mapping function that can predict the output for new, unseen data.
- **Analogy:** Learning with a teacher.
- **Types:**
    - **Classification:** The output variable is a category (e.g., "spam" or "not spam", "cat" or "dog").
    - **Regression:** The output variable is a continuous value (e.g., price, temperature).
- **Examples:** Linear Regression, Logistic Regression, Support Vector Machines (SVM), Decision Trees, Neural Networks.

### b) Unsupervised Learning
- **Concept:** The algorithm learns from an unlabeled dataset, identifying patterns and structures within the data without any predefined output variables.
- **Analogy:** Learning without a teacher.
- **Types:**
    - **Clustering:** Grouping similar data points together (e.g., customer segmentation).
    - **Dimensionality Reduction:** Reducing the number of variables in a dataset while preserving important information (e.g., Principal Component Analysis - PCA).
    - **Association Rule Learning:** Discovering interesting relationships between variables in large datasets (e.g., market basket analysis).
- **Examples:** K-Means Clustering, Hierarchical Clustering, PCA.

### c) Reinforcement Learning
- **Concept:** An agent learns to make decisions by performing actions in an environment to maximize a cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or punishments.
- **Analogy:** Learning by doing.
- **Key Components:**
    - **Agent:** The learner or decision-maker.
    - **Environment:** The world the agent interacts with.
    - **State:** The current situation of the agent.
    * **Action:** A move the agent can make.
    * **Reward:** Feedback from the environment (positive or negative).
- **Examples:** Game playing (AlphaGo), robotics, autonomous navigation, resource management.

## 3. Machine Learning vs. Artificial Intelligence

- **Artificial Intelligence (AI)** is the broader concept of creating machines that can simulate human intelligence, such as reasoning, learning, and problem-solving.
- **Machine Learning (ML)** is a subset of AI. It is the *application* of AI that provides systems with the ability to automatically learn and improve from experience without being explicitly programmed.
- **Deep Learning (DL)** is a further subset of ML that uses multi-layered neural networks to learn from vast amounts of data.

In essence, AI is the goal, and Machine Learning is one of the primary methods for achieving that goal.
