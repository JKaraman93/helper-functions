### Classification
> Remember that LinearSVC uses **loss="squared_hinge"** by default, so if we want all 3 models to produce similar results, we need to set loss="hinge".
> Also, the SVC class uses an RBF kernel by default, so we need to set kernel="linear" to get similar results as the other two models.
> Lastly, the SGDClassifier class does not have a C hyperparameter, but it has another regularization hyperparameter called **alpha**, so we can tweak it to get similar results as the other two models.
 
 ### Decision Trees
> They don't require feature scaling or centering at all
>
> A binary Decision Tree (one that makes only binary decisions, as is the case with all trees in Scikit-Learn) will end up more or less well balanced at the end of training, with one leaf per training instance if it is trained without restrictions.
>
> A node's Gini impurity is generally lower than its parent's. However, it is possible for a node to have a higher Gini impurity than its parent, as long as this increase is more than compensated for by a decrease in the other child's impurity.
