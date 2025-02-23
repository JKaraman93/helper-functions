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
> [!NOTE]
> Batch GD’s path actually stops at the minimum, while both Stochastic GD and Mini-batch GD
continue to walk around. However, don’t forget that Batch GD takes a lot of time to take each step, and
Stochastic GD and Mini-batch GD would also reach the minimum if you used a good learning schedule.
### Regularized Models
It is important to scale the data (e.g., using a StandardScaler) before performing Ridge Regression, as it is sensitive to the scale of
the input features. This is true of most regularized models.

#### Ridge Regression
> $J(\theta)=MSE(\theta)+\dfrac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$

> RR is a regularized version of Linear Regression: a regularization term is added to the MSE. This forces the learning algorithm to not only fit
the data but also keep the model weights **as small as possible**.

> Note that the regularization term should only be added to the cost function during training. Once the model is trained, you want to use the unregularized MSE
(or the RMSE) to evaluate the model’s performance.

#### Lasso Regression
> $J(\theta)=MSE(\theta)+2\alpha\sum_{i=1}^{n}\lvert\theta_i\lvert$

>LR tends to eliminate the weights of the least **important** features. Lasso Regression automatically performs **feature selection** and outputs a sparse model (i.e., with few nonzero feature weights).

>To avoid Gradient Descent from bouncing around the optimum at the end when using Lasso, you need to gradually reduce the learning
rate during training (it will still bounce around the optimum, but the steps will get smaller and smaller, so it will converge).

#### Elastic Net

> $J(\theta)=MSE(\theta)+r(2\alpha\sum_{i=1}^{n}\lvert\theta_i\lvert) +(1-r)(\dfrac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2)$

It is almost always preferable to have at least a little bit of regularization, so generally you should avoid
plain Linear Regression. Ridge is a good default, but if you suspect that only a few features are useful, you
should prefer Lasso or Elastic Net because they tend to reduce the useless features’ weights down to zero.
In general, Elastic Net is preferred over Lasso because Lasso may behave erratically **when the number of features is greater than the number of training instances or when several features are strongly correlated**.
