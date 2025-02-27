### Classification
> Remember that LinearSVC uses **loss="squared_hinge"** by default, so if we want all 3 models to produce similar results, we need to set loss="hinge".
> Also, the SVC class uses an RBF kernel by default, so we need to set kernel="linear" to get similar results as the other two models.
> Lastly, the SGDClassifier class does not have a C hyperparameter, but it has another regularization hyperparameter called **alpha**, so we can tweak it to get similar results as the other two models.
