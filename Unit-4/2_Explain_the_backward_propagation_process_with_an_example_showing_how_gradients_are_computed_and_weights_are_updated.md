# Explaining the Backward Propagation Process with an Example

Backward propagation, or backpropagation, is the core algorithm used to train artificial neural networks. It's a method for efficiently calculating the gradients of the loss function with respect to the weights and biases of the network. These gradients are then used to update the network's parameters, minimizing the error between predicted and actual outputs.

The process involves two main phases: a **forward pass** and a **backward pass**.

## 1. Forward Pass

In the forward pass, the input data is fed through the network, layer by layer, to produce an output.

*   **Input Layer**: Receives the raw input data.
*   **Hidden Layers**: Each neuron in a hidden layer computes a weighted sum of its inputs from the previous layer, adds a bias, and then applies an activation function.
    $$ Z_j = \sum_i (W_{ij} \cdot A_i) + B_j $$
    $$ A_j = f(Z_j) $$
    Where $Z_j$ is the weighted sum, $W_{ij}$ is the weight, $A_i$ is the activation from the previous layer, $B_j$ is the bias, and $f$ is the activation function.
*   **Output Layer**: The final layer produces the network's prediction.

## 2. Backward Pass (Backpropagation)

After the forward pass, the network's output is compared to the true target, and an error (loss) is calculated. The backward pass then propagates this error back through the network to compute gradients and update weights.

### Steps of the Backward Pass:

1.  **Calculate Output Layer Error ($\delta$):**
    The error is first calculated for each neuron in the output layer. This involves computing the derivative of the loss function with respect to the output of each neuron, multiplied by the derivative of its activation function.
    For a neuron $k$ in the output layer, if using Mean Squared Error (MSE) and a sigmoid activation function:
    $$ \delta_k = (A_k - Y_k) \cdot A_k (1 - A_k) $$
    Where $A_k$ is the predicted output (activation) of neuron $k$, and $Y_k$ is the true target.

2.  **Propagate Error to Hidden Layers:**
    The error is then propagated backward to the hidden layers. For each neuron $j$ in a hidden layer, its error term ($\delta_j$) is calculated based on the error terms of the neurons it connects to in the subsequent layer, weighted by the connections between them. This is where the chain rule is applied.
    $$ \delta_j = \left( \sum_k W_{jk} \cdot \delta_k \right) \cdot A_j (1 - A_j) $$
    (Again, assuming a sigmoid activation function for $A_j(1-A_j)$ as its derivative).

3.  **Compute Gradients for Weights and Biases:**
    Once the error terms ($\delta$) for all neurons are known, the gradient of the loss function with respect to each weight ($W_{ij}$) and bias ($B_j$) can be computed.
    *   **Weight Gradient**: The gradient for a weight connecting neuron $i$ (from the previous layer) to neuron $j$ (in the current layer) is the product of the error term of neuron $j$ and the activation of neuron $i$.
        $$ \frac{\partial L}{\partial W_{ij}} = \delta_j \cdot A_i $$
    *   **Bias Gradient**: The gradient for a bias of neuron $j$ is simply its error term.
        $$ \frac{\partial L}{\partial B_j} = \delta_j $$

4.  **Update Weights and Biases:**
    Finally, an optimization algorithm (e.g., Gradient Descent) uses these gradients to adjust the weights and biases. The parameters are updated in the direction that reduces the loss, scaled by a learning rate ($\alpha$).
    $$ W_{ij}^{\text{new}} = W_{ij}^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial W_{ij}} $$
    $$ B_j^{\text{new}} = B_j^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial B_j} $$

This entire cycle (forward pass, loss, backward pass, update) is repeated for multiple iterations (epochs) and typically on mini-batches of data until the network's performance converges.

---

## Example: Simple Neural Network with One Hidden Layer

Let's consider a very simple neural network with:
*   2 input neurons ($x_1, x_2$)
*   1 hidden layer with 2 neurons ($h_1, h_2$)
*   1 output neuron ($o_1$)
*   Sigmoid activation function for all neurons.
*   Mean Squared Error (MSE) as the loss function.

**Given:**
*   Input: $X = [0.05, 0.10]$
*   Target Output: $Y = [0.01, 0.99]$ (for two output neurons, but for simplicity, let's assume one output neuron with target $Y = 0.75$)
*   Initial Weights and Biases (example values):
    *   $W_{11}, W_{12}, W_{21}, W_{22}$ (input to hidden)
    *   $W_{31}, W_{41}$ (hidden to output)
    *   $B_1, B_2$ (hidden biases)
    *   $B_3$ (output bias)
*   Learning Rate ($\alpha$) = 0.5

### Forward Pass:

1.  **Hidden Layer Calculation:**
    *   $Z_{h1} = (X_1 \cdot W_{11}) + (X_2 \cdot W_{21}) + B_1$
    *   $A_{h1} = \text{sigmoid}(Z_{h1})$
    *   $Z_{h2} = (X_1 \cdot W_{12}) + (X_2 \cdot W_{22}) + B_2$
    *   $A_{h2} = \text{sigmoid}(Z_{h2})$

2.  **Output Layer Calculation:**
    *   $Z_{o1} = (A_{h1} \cdot W_{31}) + (A_{h2} \cdot W_{41}) + B_3$
    *   $A_{o1} = \text{sigmoid}(Z_{o1})$ (This is our predicted output)

### Backward Pass:

1.  **Calculate Output Layer Error ($\delta_{o1}$):**
    Let's assume $Y_{target} = 0.75$ and $A_{o1} = 0.70$ (after forward pass).
    Loss (MSE) = $\frac{1}{2} (Y_{target} - A_{o1})^2$
    $$ \frac{\partial L}{\partial A_{o1}} = -(Y_{target} - A_{o1}) $$
    Derivative of sigmoid: $f'(Z) = A(1-A)$
    $$ \delta_{o1} = \frac{\partial L}{\partial A_{o1}} \cdot f'(Z_{o1}) = -(Y_{target} - A_{o1}) \cdot A_{o1}(1 - A_{o1}) $$
    $$ \delta_{o1} = -(0.75 - 0.70) \cdot 0.70(1 - 0.70) = -0.05 \cdot 0.70 \cdot 0.30 = -0.0105 $$

2.  **Calculate Gradients for Weights to Output Layer ($W_{31}, W_{41}$):**
    $$ \frac{\partial L}{\partial W_{31}} = \delta_{o1} \cdot A_{h1} $$
    $$ \frac{\partial L}{\partial W_{41}} = \delta_{o1} \cdot A_{h2} $$
    (Assuming $A_{h1}=0.59$ and $A_{h2}=0.60$ from forward pass)
    $$ \frac{\partial L}{\partial W_{31}} = -0.0105 \cdot 0.59 \approx -0.006195 $$
    $$ \frac{\partial L}{\partial W_{41}} = -0.0105 \cdot 0.60 \approx -0.0063 $$

3.  **Update Weights to Output Layer:**
    $$ W_{31}^{\text{new}} = W_{31}^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial W_{31}} $$
    $$ W_{41}^{\text{new}} = W_{41}^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial W_{41}} $$
    (e.g., if $W_{31}^{\text{old}} = 0.40$, then $W_{31}^{\text{new}} = 0.40 - 0.5 \cdot (-0.006195) = 0.4030975$)

4.  **Calculate Hidden Layer Error ($\delta_{h1}, \delta_{h2}$):**
    This error depends on the error of the output neuron it connects to.
    $$ \delta_{h1} = (W_{31} \cdot \delta_{o1}) \cdot A_{h1}(1 - A_{h1}) $$
    $$ \delta_{h2} = (W_{41} \cdot \delta_{o1}) \cdot A_{h2}(1 - A_{h2}) $$
    (e.g., if $W_{31}=0.40$, $W_{41}=0.45$)
    $$ \delta_{h1} = (0.40 \cdot -0.0105) \cdot 0.59(1 - 0.59) \approx -0.0042 \cdot 0.59 \cdot 0.41 \approx -0.00101 $$
    $$ \delta_{h2} = (0.45 \cdot -0.0105) \cdot 0.60(1 - 0.60) \approx -0.004725 \cdot 0.60 \cdot 0.40 \approx -0.001134 $$

5.  **Calculate Gradients for Weights to Hidden Layer ($W_{11}, W_{12}, W_{21}, W_{22}$):**
    $$ \frac{\partial L}{\partial W_{11}} = \delta_{h1} \cdot X_1 $$
    $$ \frac{\partial L}{\partial W_{21}} = \delta_{h1} \cdot X_2 $$
    $$ \frac{\partial L}{\partial W_{12}} = \delta_{h2} \cdot X_1 $$
    $$ \frac{\partial L}{\partial W_{22}} = \delta_{h2} \cdot X_2 $$
    (e.g., if $X_1=0.05, X_2=0.10$)
    $$ \frac{\partial L}{\partial W_{11}} = -0.00101 \cdot 0.05 \approx -0.0000505 $$

6.  **Update Weights to Hidden Layer:**
    $$ W_{11}^{\text{new}} = W_{11}^{\text{old}} - \alpha \cdot \frac{\partial L}{\partial W_{11}} $$
    And similarly for $W_{12}, W_{21}, W_{22}$ and all biases.

This example illustrates how the error signal is systematically propagated backward, and how each weight's contribution to the overall error is calculated and used for adjustment. This iterative process allows the network to learn and improve its predictions over time.
