## Tqdm
```
from tqdm.notebook import tqdm, trange

n_epochs = 5 
n_steps = 20
with trange(1, n_epochs + 1, desc="All epochs") as epochs:
  for epoch in epochs:
    with trange(1, n_steps + 1, desc=f"Epoch {epoch}/{n_epochs}") as steps:
      for step in steps:
        steps.set_postfix(status)  # add extra info: `status` is a dictionary which includes loss, accuracy etc.
        time.sleep(1.0)
```
