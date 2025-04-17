> [! TIP]
> . It can also help to increase the dropout rate for large layers, and reduce it for small ones. It can also help to increase the dropout rate for its inputs.

### 1Cycle 

Phase |	Learning Rate (LR)	| Momentum 
--- | --- | --- 
Start	|Low LR (η₀)	|High momentum (e.g., 0.95) 
First |half	LR increases linearly to η₁	|Momentum decreases linearly to a low value (e.g., 0.85)
Second |half	LR decreases back to η₀	|Momentum increases back to the high value
Final |few epochs	LR drops much further (by orders of magnitude)	|Momentum remains high

- Warm-up phase (low to high LR):
  >Helps escape poor local minima or saddle points.\
  >Gradually adapts weights to learning.

- Cooldown phase (high to low LR):
    >Stabilizes learning as training progresses. \
    >Encourages convergence to a minimum.

- Final sharp drop:

    >Improves generalization by refining the model with very small updates. \
    >Similar to simulated annealing.

- Momentum acts inversely:
    >Helps to smooth out large updates early and small ones later.
```
class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_of_epoch_losses = 0

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]  # the epoch's mean loss so far 
        new_sum_of_epoch_losses = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_of_epoch_losses - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epoch_losses
        lr = self.model.optimizer.learning_rate.numpy()
        self.rates.append(lr)
        self.losses.append(batch_loss)
        self.model.optimizer.learning_rate = lr * self.factor
```
### Performance scheduling 
```
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

### Power scheduling        $\ \eta(t) = \frac{\eta_0} {(1 + \frac{t}{s})^c}$
```
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.01,
    decay_steps=10_000,
    decay_rate=1.0,
    staircase=False)
```

### Exponential Scheduling  $\ \eta(t) = \eta_0 0.1^\frac{t}{s}$
```
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
```
