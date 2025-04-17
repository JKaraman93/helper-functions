## Dropout
> [!TIP]
> It can also help to increase the dropout rate for large layers, and reduce it for small ones. It can also help to increase the dropout rate for its inputs.
>
> In practice, you can usually apply dropout only to the neurons in the top one to three layers (excluding the output layer).

### Mc Dropout 
Using Monte Carlo Dropout, we perform multiple stochastic forward passes (e.g., 100) over the test set with dropout enabled at inference time. We then average the predictions to estimate the final output. These predictions are not the same each time due to the randomness introduced by dropout.
