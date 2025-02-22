> Standardization (StandardScaler) is crucial for very high-degree polynomials to avoid numerical instability.

### Linear Regression (LR)
#### Normal Equation (NE) $\hat\theta=(X^TX)^{-1}X^Ty$
Approach|Order of Magnitude
---|:---:
Normal Eq|$\mathcal{O}(n^{2.4})$ to $\mathcal{O}(n^{3})$ 
SVD|$\mathcal{O}^{2}$

>Both the Normal Equation and the SVD approach get very slow when the number of features grows large. On the positive side, both are linear with regard to the number of instances in the training set, so they handle large training
sets efficiently, provided they can fit in memory.

#### Gradient Descent (GD)
> All features should have a similar scale **(StandardScaler)**, or else it will take much longer to converge.

> GD scales well with the number of features; training a LR model when there are hundreds of thousands of features is much faster using GD than using the NE or SVD decomposition.

1. Batch Gradient Descent (Full)
2. Stochastic Gradient Descent
> When using Stochastic Gradient Descent, the training instances must be independent and identically distributed (IID) to ensure that the parameters get pulled toward the global optimum, on average. A simple way to ensure this is to shuffle the instances during training
4. Mini-Batch Gradient Descent
