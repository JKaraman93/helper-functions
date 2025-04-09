## Optimizers

### Gradient Descent
  $\theta \leftarrow \theta - \eta\nabla_\theta J(\theta)$

### Momentum Optimizer
  $m \leftarrow \beta m - \eta\nabla_\theta J(\theta)$ \
  $\theta \leftarrow \theta + m$ | β, called the momentum, which must be set between 0 (high friction) and 1 (no friction) (default 0.9).
      
    In deep neural networks that don’t use Batch Normalization, the upper layers will often end up having 
    inputs with very different scales, so using momentum optimization helps a lot.
    
### Nesterov Accelerated Gradient
  $m \leftarrow \beta m - \eta\nabla_\theta J(\theta + \beta m)$ \
  $\theta \leftarrow \theta + m$
