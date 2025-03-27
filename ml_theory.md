### Perceptron learning rule
$w_{i,j}^{(next\ step)} = w_{i,j} + \eta(y_j - \hat{y}_j)x_i$

> [!NOTE]
> Contrary to Logistic Regression classifiers, Perceptrons do not output a class
probability. This is one reason to prefer Logistic Regression over Perceptrons.
Moreover, **Perceptrons do not use any regularization by default**, and training stops as
soon as there are no more prediction errors on the training set, so the model typically**does not generalize** as well as Logistic Regression or a linear SVM classifier. However,
Perceptrons may train a bit faster.

>The **decision boundary** is defined where $y^=0$, which is the hyperplane:\
$w_1x_1+w_2x_2+ .. +w_nx_n+b=0$
    
#### Loss functions

| Functions | Target Class | Activation |
| -------- | ------- | ------- |
| Cross-Entropy Loss | One-hot encoded labels (e.g., [0,1,0])| Softmax |
| Sparse Cross-Entropy Loss	| Integer labels (e.g., 1 for "cat") | Softmax |

