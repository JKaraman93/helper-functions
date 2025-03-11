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
### Anomaly detection

### Density estimation
