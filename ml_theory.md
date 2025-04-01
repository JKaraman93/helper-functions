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

### Vanishing Gradients

    Gradient values get smaller as they propagate backward through layers (especially in deep networks).
    Lower (earlier) layers receive very small gradients → Their weights update very slowly
    Many activation functions (like sigmoid or tanh) have small derivatives for extreme values.
    
✅ Use activation functions with better gradients:

    ReLU (Rectified Linear Unit) instead of Sigmoid or Tanh.

    Leaky ReLU, Parametric ReLU (PReLU) for better flow of gradients.

✅ Batch Normalization:

    Keeps activations well-scaled.

    Prevents gradients from shrinking too much.

✅ Use Residual Networks (ResNets):

    Includes skip connections to pass information directly to later layers.

    Helps gradients flow more easily.

✅ Use Proper Weight Initialization:

    Xavier/Glorot Initialization: Works well for Sigmoid/Tanh.

    He Initialization: Works well for ReLU-based networks.

>[!TIP]
> When the values to predict can vary by many orders of magnitude, you may want to predict the **logarithm** of the target value rather than the target value directly (e.g Some houses cost $50,000, others $5,000,000, the scale difference is huge). Simply computing the **exponential** of the neural network's output will give you the estimated value (since exp(log v) = v).
