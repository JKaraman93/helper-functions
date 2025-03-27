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

| Classification Type	| # of Output Neurons	| Activation Function	| Loss Function |	Target Labels Format
| -------- | :-------: | ------- | ------- | ------- |
| Binary Classification	| 1		| Sigmoid (σ(z))	| 	Binary Cross-Entropy 	| Binary (0 or 1)
| Multi-Class (C classes, One-Hot Labels)		| C		| Softmax		| Categorical Cross-Entropy | One-hot encoded vectors (e.g., [0,1,0,0] for class 1)
|Multi-Class (C classes, Integer Labels)		| C		| Softmax		| Sparse Categorical Cross-Entropy | Integer class labels (e.g., 1, 2, 3)
Multi-Label Classification (each sample can belong to multiple classes)	| C	| Sigmoid (applied independently to each neuron)| Binary Cross-Entropy| Binary vector for each sample (e.g., [1,0,1] for belonging to classes 0 and 2)
| Ordinal Classification (Classes with an order, e.g., "bad" < "okay" < "good")	| 1	| Sigmoid (or Softmax with ordinal encoding)| Custom loss (e.g., Mean Squared Error or Ordinal Cross-Entropy)| Ordered integer labels (0,1,2,...)

#### Regularization technique
> [!TIP]
> The **auxiliary output** acts as a separate learning signal, ensuring that even earlier layers contribute directly to predictions.\
> This improves <ins>generalization</ins> by forcing different parts of the network to be useful independently
