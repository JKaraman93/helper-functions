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
