## Unsupervised Learning

### Clustering
#### Dimensionality reduction
Once a dataset has been clustered, it is usually possible to measure each instance’s affinity with each cluster: affinity is any measure of how well an instance fits into a cluster. 
Each instance’s feature vector x can then be replaced with the vector of its cluster affinities.

#### For Feature Engineering
The cluster affinities can often be useful as extra features.

#### For anomaly detection (also called outlier detection)
Any instance that has a low affinity to all the clusters is likely to be an anomaly.

#### For semi-supervised learning
If you only have a few labels, you could perform clustering and propagate

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

### Anomaly detection

### Density estimation
