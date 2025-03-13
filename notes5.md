## Unsupervised Learning

### Clustering
**Dimensionality reduction:** Once a dataset has been clustered, it is usually possible to measure each instance’s affinity with each cluster: affinity is any measure of how well an instance fits into a cluster. Each instance’s feature vector x can then be replaced with the vector of its cluster affinities.

**Feature Engineering**: The cluster affinities can often be useful as extra features.

**Anomaly detection (also called outlier detection)**: Any instance that has a low affinity to all the clusters is likely to be an anomaly.

**Semi-supervised learning**
If you only have a few labels, you could perform clustering and propagate
1) Apply clustering to the full training set
2) Label the representative points of each cluster
3) Propagate the labels to all the other instances
4) (Optional) Ignore the 1% instances that are farthest from their cluster center (Outliers)

>[!NOTE]
> <ins> Active learning - Uncertainty sampling </ins>
> 1. The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled
instances.
> 2. The instances for which the model is most uncertain (i.e., when its
estimated probability is lowest) are given to the expert for labeling.
> 3. You iterate this process until the performance improvement stops
being worth the labeling effort.

### Kmeans
> [!IMPORTANT]
> Moreover, K-Means does not behave very well when the clusters have varying sizes, different densities, or nonspherical
shapes.
>
>  It is important to **scale** the input features before you run K-Means, or the clusters may
be very stretched and K-Means will perform poorly.
```
  ### KMeans SEGMENTATION  ###
  kmeans = KMeans(n_clusters=c, n_init=10, random_state=42).fit(X)
  segmented_img = kmeans.cluster_centers_[kmeans.labels_]
  segmented_img = segmented_img.reshape(image.shape)
  plt.imshow(segmented_img/255.)
```
### DBSCAN
If the density varies significantly across the clusters, or if there’s no sufficiently low-density region around some clusters, 
DBSCAN can struggle to capture all the clusters properly. Moreover, it does not scale well to large datasets.

### Agglomerative clustering
It can scale nicely to large numbers of instances if you provide a **connectivity matrix**, which is a
sparse m × m matrix that indicates which pairs of instances are neighbors (e.g., returned by
sklearn.neighbors.kneighbors_graph()). Without a connectivity matrix, the algorithm **does not scale well to large datasets**

### BIRCH
was designed specifically for very large datasets,and it can be faster than batch K-Means, without having to store all the instances in the
tree: this approach allows it to use limited memory, while handling huge datasets.

### Spectral clustering 
This algorithm takes a similarity matrix between the instances and creates a low-dimensional embedding from it (i.e., it reduces the
matrix’s dimensionality), then it uses another clustering algorithm in this low-dimensional space

### Gaussian Mixture Model
A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian
distributions whose parameters are unknown. All the instances generated from a single Gaussian distribution form a cluster that typically looks like
an ellipsoid. Each cluster can have a different ellipsoidal shape, size, density, and orientation.

Expectation-Maximization (EM) algorithm, which has many similarities with the K-Means algorithm: it also initializes the cluster
parameters randomly, then it repeats two steps until convergence, first assigning instances to clusters (this is called the expectation step) and then
updating the clusters (this is called the maximization step).

A Gaussian mixture model is a generative model.
> [!TIP]
> Gaussian mixture models try to fit all the data, including the outliers, so if you have too
many of them, this will bias the model’s view of “normality,” and some outliers may
wrongly be considered as normal. If this happens, you can try to fit the model once, use
it to detect and remove the most extreme outliers, then fit the model again on the
cleaned-up dataset.
> **BayesianGaussianMixture** class, which is capable of giving weights equal (or close) to zero to unnecessary clusters.

> [!NOTE]
> <ins> Dimensionality reduction  - inverse_transform() method  </ins>
>
> If you compare the reconstruction error of a normal instance with the
reconstruction error of an anomaly, the latter will usually be much
larger.

