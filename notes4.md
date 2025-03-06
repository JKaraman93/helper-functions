### The Curse of Dimensionality

> High-dimensional datasets are at risk of being very sparse: most
training instances are likely to be far away from each other. This also means
that a new instance will likely be far away from any training instance,
making predictions much less reliable than in lower dimensions, since they
will be based on much larger extrapolations. In short, the more dimensions
the training set has, the greater the risk of **overfitting** it.
>
> ### PCA
>
> It assumes that the dataset is centered around the origin.

> $X_{d-proj}=XW_d$ where $W_d$, defined as the matrix containing the first $d$ columns of $V$ (PCA ccomponents matrix)
