# Parameter Estimation and Maximum Likelihood Estimation (MLE)

## What is Parameter Estimation?

**Parameter estimation** is a fundamental process in statistics and machine learning where the goal is to estimate the unknown parameters of a probability distribution or a model based on observed data. In many real-world scenarios, we assume that our data is generated from a certain underlying process or distribution (e.g., Gaussian, Bernoulli, Poisson), but the specific values of the parameters governing that distribution are unknown. Parameter estimation seeks to find the "best" values for these parameters that make the observed data most probable or best fit the model.

For example:
*   If we assume a dataset follows a normal distribution, we need to estimate its mean (\(\mu\)) and standard deviation (\(\sigma\)).
*   In a linear regression model, we need to estimate the coefficients (weights) that define the relationship between independent and dependent variables.
*   In a coin toss experiment, we might want to estimate the probability of getting a head (\(p\)).

## Maximum Likelihood Estimation (MLE)

**Maximum Likelihood Estimation (MLE)** is a widely used and powerful method for parameter estimation. The core idea behind MLE is to find the values of the model parameters that maximize the **likelihood function**. The likelihood function quantifies how probable it is to observe the given dataset under different possible values of the model parameters. In simpler terms, MLE asks: "Given the data we have observed, what parameter values make this data most likely to occur?"

### The Likelihood Function

Let \(D = \{x_1, x_2, \dots, x_n\}\) be a set of \(n\) independent and identically distributed (i.i.d.) observations from a probability distribution with an unknown parameter \(\theta\). The probability density function (PDF) or probability mass function (PMF) of a single observation \(x_i\) given \(\theta\) is \(P(x_i | \theta)\).

The **likelihood function** \(L(\theta | D)\) is defined as the product of the probabilities of observing each data point, given the parameter \(\theta\):
$$ L(\theta | D) = \prod_{i=1}^{n} P(x_i | \theta) $$ 

### The Log-Likelihood Function

To simplify calculations, especially when dealing with products and derivatives, it is common practice to work with the **log-likelihood function**, which is the natural logarithm of the likelihood function:
$$ \log L(\theta | D) = \sum_{i=1}^{n} \log P(x_i | \theta) $$ 
Maximizing the log-likelihood is equivalent to maximizing the likelihood, because the logarithm is a monotonically increasing function.

### Steps for MLE:

1.  **Formulate the Likelihood Function**: Write down the likelihood function for the observed data given the assumed probability distribution and its parameters.
2.  **Take the Log-Likelihood**: Convert the likelihood function into its logarithmic form to simplify differentiation.
3.  **Differentiate and Set to Zero**: Calculate the partial derivative of the log-likelihood function with respect to each parameter and set these derivatives to zero.
4.  **Solve for Parameters**: Solve the resulting equations to find the values of the parameters that maximize the log-likelihood. These values are the Maximum Likelihood Estimates (\(\hat{\theta}\)).
5.  **Verify Maximum**: (Optional but good practice) Use second derivatives to confirm that the found values correspond to a maximum, not a minimum or saddle point.

## Example: Estimating the Probability of Heads in a Coin Toss

Let's say we have a biased coin, and we want to estimate the probability of getting a head (\(p\)). We perform \(n\) coin tosses, and we observe \(k\) heads and \(n-k\) tails.

1.  **Assumed Distribution**: Each coin toss is a Bernoulli trial. The number of heads in \(n\) tosses follows a Binomial distribution.
    The probability of observing \(k\) heads in \(n\) tosses, given \(p\), is:
    $$ P(k \text{ heads in } n \text{ tosses} | p) = \binom{n}{k} p^k (1-p)^{n-k} $$ 

2.  **Likelihood Function**: For our observed data (e.g., \(k\) heads in \(n\) tosses), the likelihood function is:
    $$ L(p | k, n) = \binom{n}{k} p^k (1-p)^{n-k} $$ 

3.  **Log-Likelihood Function**:
    $$ \log L(p | k, n) = \log \binom{n}{k} + k \log(p) + (n-k) \log(1-p) $$ 

4.  **Differentiate and Set to Zero**: We want to find \(p\) that maximizes this. We differentiate with respect to \(p\):
    $$ \frac{d}{dp} \log L(p | k, n) = \frac{k}{p} - \frac{n-k}{1-p} $$ 
    Set the derivative to zero:
    $$ \frac{k}{p} - \frac{n-k}{1-p} = 0 $$ 
    $$ \frac{k}{p} = \frac{n-k}{1-p} $$ 
    $$ k(1-p) = p(n-k) $$ 
    $$ k - kp = pn - pk $$ 
    $$ k = pn $$ 
    $$ \hat{p} = \frac{k}{n} $$ 

5.  **Result**: The Maximum Likelihood Estimate for the probability of getting a head (\(\hat{p}\)) is simply the observed frequency of heads (\(k\)) divided by the total number of tosses (\(n\)). This intuitively makes sense: if you get 7 heads in 10 tosses, your best estimate for the probability of heads is 0.7.

## Conclusion

Maximum Likelihood Estimation is a powerful and widely used method for parameter estimation due to its desirable statistical properties (e.g., consistency, asymptotic efficiency). It provides a systematic way to infer the parameters of a model that best explain the observed data, forming a cornerstone of statistical inference and machine learning.
