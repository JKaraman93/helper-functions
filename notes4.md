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

> $X_{recovered}=X_{d-proj}W_d^T$ - PCA for Compression

### Random Projection

> Random Projection algorithm projects the data to
a lower-dimensional space using a random linear projection. This may
sound crazy, but it turns out that such a random projection is actually very
likely to preserve distances fairly well.
>
> It only relies on **m** and **ε** and not on n | n = #features m = #instances,  ε = error tolerance
>
> it’s usually preferable to use **SparseRandomProjection** transformer instead of the first one, especially for **large or sparse datasets**.
>
> LLE is quite different from the projection techniques, and
it’s significantly more complex, but it can also perform much better,
especially if the data is non-linear.

### Locally Linear Embedding (LLE) - (Manifold Learning)
> It works by first measuring how each training instance linearly relates to its
nearest neighbors, and then looking for a low-dimensional representation of
the training set where these local relationships are best preserved.
>
> LLE is quite different from the projection techniques, and
it’s significantly more complex, but it can also perform much better,
especially if the data is non-linear.
>
> More specifically, it
tries to find the weights wi,j such that
> $\hat{W} = argmin_m \sum_{i=1}^{m}(x^{(i))}$
> is as small as possible, assuming wi,j = 0 if x(j) is not one of
the k nearest neighbors of x(i).

The second
step is to map the training instances into a d-dimensional space (where d <
n) while preserving these local relationships as much as possible. If z(i) is
the image of x(i) in this d-dimensional space, then we want the squared
distance between z(i) and ∑ ŵ z to be as small as possible
