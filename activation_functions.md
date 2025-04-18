## Summary

**Relu** is usually a good default for the hidden layers, as it is fast and yields good results.

The **leaky ReLU** variants of ReLU can **improve** the model's quality without hindering its speed too much compared to ReLU.  

For **large** neural nets and more **complex** problems, **GLU, Swish and Mish** can give you a slightly higher quality model, but they have a computational cost. 

The **hyperbolic tangent (tanh)** can be useful in the output layer if you need to output a number in a fixed range (by default between –1 and 1), but nowadays it is not used much in hidden layers, except in recurrent nets.  

The **sigmoid** activation function is also useful in the output layer when you need to estimate a probability (e.g., **for binary classification**), but it is rarely used in hidden layers (there are exceptions—for example, for the coding layer of variational autoencoders).

The **softplus** activation function is useful in the output layer when you need to ensure that the output will **always be positive**.  

The **softmax** activation function is useful in the output layer to estimate probabilities for **mutually exclusive classes**, but it is rarely (if ever) used in hidden layers.
