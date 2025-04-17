### Number of Layers

👉 When to increase layers?

    Complex tasks (e.g., image recognition, NLP, speech processing) → Deep networks capture hierarchical features.

    High-dimensional input data (e.g., images, text embeddings) → More layers help extract meaningful representations.

👉 When to decrease layers?

    Simple tasks (e.g., linear classification) → Too many layers lead to overfitting.

    Small datasets → Deep networks memorize rather than generalize.

🔧 Tuning strategy:

    Start with 2-3 layers, increase gradually while monitoring validation loss.

    Use GridSearch or Bayesian Optimization to test different architectures.

### Number of Neurons per Layer

👉 When to increase neurons?

    Complex patterns in data → More neurons can capture richer representations.

    Underfitting (high bias, low accuracy) → The model lacks capacity.

👉 When to decrease neurons?

    Overfitting (high variance, training accuracy ≫ validation accuracy) → Too many neurons memorize noise.

    Latency-sensitive applications → Fewer neurons reduce computation time.

🔧 Tuning strategy:

    Start with (input features × 2) neurons in the first layer.

    Try powers of 2 (e.g., 32, 64, 128, 256) and monitor performance.

    Use pruning (reduce neurons layer by layer if overfitting occurs).

### Learning Rate (LR)

👉 When to increase LR?

    Training is too slow → If loss decreases too gradually, increasing LR speeds up convergence.

    Large datasets → Larger LR is effective since noise cancels out bad updates.

👉 When to decrease LR?

    Diverging loss (spikes up and down) → LR is too high, making updates unstable.

    Complex models (deep networks) → Smaller LR prevents skipping local minima.

🔧 Tuning strategy:

    Start with 0.01 for SGD, 0.001 for Adam.

    Use LR scheduling: Decrease LR after plateaus.

    Use Cyclical LR or One-Cycle Policy for efficient training.

### Optimizer Choice

👉 SGD (with momentum)

    Good for large datasets (e.g., ImageNet training).

    Can get stuck in local minima.

    Works well with a well-tuned LR schedule.

👉 Adam

    Good for small datasets and NLP (handles noisy gradients well).

    Sometimes generalizes poorly compared to SGD.

👉 RMSprop

    Works well in recurrent networks (RNNs, LSTMs).

    Effective for unstable loss surfaces.

🔧 Tuning strategy:

    Start with Adam for most cases, SGD with momentum for large datasets.

    Experiment with learning rate decay.

### Batch Size

👉 When to increase batch size?

    Faster training → Larger batches improve GPU utilization.

    BatchNorm present → Works better with large batch sizes.

👉 When to decrease batch size?

    Limited memory (large models, images) → Smaller batches avoid OOM errors.

    Generalization issues → Small batches introduce noise, helping models escape sharp local minima.

🔧 Tuning strategy:

    Use 32 or 64 as a starting point.

    Large datasets: 128, 256, 512.

    Small datasets: 8, 16, 32.

### Activation Function

👉 ReLU

    Best for deep networks → Avoids vanishing gradient.

    Can suffer from dead neurons (dying ReLU problem).

👉 LeakyReLU / Parametric ReLU

    Fixes dead ReLU problem.

    Used in GANs and deeper models.

👉 Tanh

    Works better for small networks.

    Can still suffer from vanishing gradients.

👉 Sigmoid

    Only for binary classification.

    Avoid in deep networks (vanishing gradients).

🔧 Tuning strategy:

    ReLU is the default for deep networks.

    LeakyReLU or ELU can be used for better performance.

    Sigmoid/Tanh only for small networks or output layers.

### Hyperparameter Tuning Techniques

    GridSearchCV – Works for shallow networks but too slow for deep learning.

    RandomizedSearchCV – Faster, but not always optimal.

    Bayesian Optimization – More efficient exploration.

    Hyperband / HalvingGridSearchCV – Stops bad configurations early.

    Neural Architecture Search (NAS) – Automates deep learning tuning.

### Final Recommendations
- Start simple (few layers, moderate neurons, Adam optimizer).
- Tune one hyperparameter at a time.
- Use callbacks (e.g., ReduceLROnPlateau, EarlyStopping).
- onsider AutoML tools if available (e.g., KerasTuner, Optuna).

### A Good Configuration

Hyperparameter | Default value 
 --- | --- 
|Kernel initializer | He initialization |
Activation function | ReLU if shallow; Swish if deep
Normalization | None if shallow; Batch Norm if deep
Regularization | Early stopping; Weight decay if needed
Optimizer | Nesterov Accelerated Gradients or AdamW
Learning rate schedule | Performance scheduling or 1cycle

<ins> In case of  simple stack of dense layers </ins>

Hyperparameter | Default value
 --- | --- 
Kernel initializer |LeCun initialization
Activation function | SELU
Normalization | None (self-normalization)
Regularization | Alpha dropout if needed
Optimizer |Nesterov Accelerated Gradients
Learning rate schedule | Performance scheduling or 1cycle


