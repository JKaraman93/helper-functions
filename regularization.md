
## Rationale
Large weights usually mean the model is very **sensitive to input changes** — this leads to overfitting.

By penalizing large weights, we encourage the model to find simpler, smoother solutions that generalize better to new data.

## $\mathbf{L_1}$ (Lasso)  

$Loss_{total}​=Loss_{original}​+λ\sum|w_i|$

$w_i​ \leftarrow w_i ​− \eta(\frac {\partial Loss}{\partial w_i} + 2\lambda w_i)​$  

If $w_i$​ is small, it can easily be shrunk to zero, because the **fixed-size** subtraction can outweigh its current value.  
Encourages **sparse weights** → drives many weights exactly to **zero**.
Good for **feature selection**.

## $\mathbf{L_2}$ (Ridge)   
$Loss_{total}​=Loss_{original}​+λ\sum w_i^2$  

$w_i​ \leftarrow w_i ​− \eta(\frac {\partial Loss}{\partial w_i} + \lambda sign(w_i))​$  

The gradient is **proportional** to $wi$.  
Encourages **smaller weights**, but not exactly zero.
**All features** contribute a bit to prediction.

## Dropout

> [!NOTE]
> It does slow down training, in general roughly by a factor of two. 

> [!TIP]
> It can also help to increase the dropout rate for large layers, and reduce it for small ones. It can also help to increase the dropout rate for its inputs.  
> In practice, you can usually apply dropout only to the neurons in the top one to three layers (excluding the output layer).

### Mc Dropout 
Using Monte Carlo Dropout, we perform multiple stochastic forward passes (e.g., 100) over the test set with **dropout enabled at inference time**. We then average the predictions to estimate the final output. These predictions are not the same each time due to the randomness introduced by dropout.

```
class MCDropout(tf.keras.layers.Dropout):
  def call(self, inputs, training=None):
    return super().call(inputs, training=True)
```
## Max-Norm Regularization
It constrains the weights w of the incoming connections such that $∥ w ∥_2 ≤ r$, where r is the max-norm hyperparameter and $∥ · ∥_2$ is the $ℓ_2$ norm and rescales w if needed $(w \leftarrow w \frac{r}{∥ w ∥_2})$.
```
kernel_constraint=tf.keras.constraints.max_norm(1.)) # r = 1.
```
> [!NOTE]
> The max_norm() function has an **axis** argument that defaults to 0. A
Dense layer usually has weights of shape [number of inputs, number of
neurons], so using axis=0 means that the max-norm constraint will apply
independently to each neuron’s weight vector. If you want to use max-norm
with **convolutional layers** (see Chapter 14), make sure to set the
max_norm() constraint’s axis argument appropriately (usually axis=
[0, 1, 2]).
