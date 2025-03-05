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

### Random Forest
> Ensemble methods work best when the predictors are as independent from one another as possible. One way to get diverse classifiers is to train them using very different algorithms. This increases the chance that they will make very different types of errors, improving the ensemble’s accuracy.
>
> Bootstrapping introduces a bit more diversity in the subsets that each predictor is trained
on, so bagging ends up with a slightly higher bias than pasting; but the extra diversity
also means that the predictors end up being less correlated, so the ensemble’s variance isreduced. Overall, bagging often results in better models, which explains why it is
generally preferred

### AdaBoost 
> The algorithm first trains a base
classifier (such as a Decision Tree) and uses it to make predictions on the training set.
The algorithm then increases the relative weight of misclassified training instances.
Then it trains a second classifier, using the updated weights, and again makes
predictions on the training set, updates the instance weights

### Gradient Boosting 
> Just like AdaBoost,
Gradient Boosting works by sequentially adding predictors to an ensemble, each one
correcting its predecessor. However, instead of tweaking the instance weights at every
iteration like AdaBoost does, this method tries to fit the new predictor to the **residual
errors** made by the previous predictor.
```
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)
## Prediction 
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

> The **learning_rate** hyperparameter scales the contribution of each tree. If you set it
to a low value, such as 0.05, you will need more trees in the ensemble to fit the
training set, but the predictions will usually generalize better.
> 
> if you set the **n_iter_no_change** hyperparameter to an integer value, say 10, then theGradientBoostingRegressor will automatically stop adding more trees during
training if it sees that the last 10 trees didn’t help. This is simply *early stopping*
> 
> GradientBoostingRegressor class also supports a **subsample**
hyperparameter, which specifies the fraction of training instances to be used for training
each tree. For example, if subsample=0.25, then each tree is trained on 25% of the
training instances, selected randomly (Stochastic Gradient Boosting).

### Additional Algorithms
> Several other optimized implementations of Gradient Boosting are available in the Python ML
ecosystem, in particular: XGBoost, CatBoost, and LightGBM. These libraries have been around for
several years, they are all specialized for Gradient Boosting, their APIs are very similar to Scikit-
Learn’s, and they provide many additional features, including GPU-acceleration: you should definitely
check them out! Moreover, there’s a newcomer in the forest arena: TensorFlow Random Forests was
released in 2021, and it provides optimized implementations of many Random Forest algorithms: plain
Random Forests, Extra Trees, GBRT, and several more.
