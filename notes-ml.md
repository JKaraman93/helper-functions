### Cross-Validation Stratified
```
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, suffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
  clone_clf = clone(sgd_clf)
  X_train_folds = X_train[train_index]
  y_train_folds = y_train_5[train_index]
  X_test_fold = X_train[test_index]
  y_test_fold = y_train_5[test_index]

  clone_clf.fit(X_train_folds, y_train_folds)
  y_pred = clone_clf.predict(X_test_fold)
  n_correct = sum(y_pred == y_test_fold)
  print(n_correct / len(y_pred))
```
### Preprocessing Pipeline
```
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler()) ])

cat_pipeline = Pipeline([ #('ordinal_encoder', OrdinalEncoder()),
                          ('imputer', SimpleImputer(strategy='most_frequent')),
                          ('cat_encoder',OneHotEncoder(sparse_output=False))  ])

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
     ('cat', cat_pipeline, cat_attribs) ],
remainder='drop')

X_train = preprocess_pipeline.fit_transform(train_data)
X_train = pd.DataFrame(X_train, columns=preprocess_pipeline.get_feature_names_out(), index= train_data.index)
```
### Get all files located in a dir
```
ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
```
### Email parser 
```
import email
import email.policy

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
```
### various packages
```
from sklearn.metrics import confuion_matrix, ConfusionMatrixDisplay, precision_recall_curve, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
```
### Countour plot
```
xc = np.linspace(0,7, 500)
yc= np.linspace(0, 3.5, 200)

x0, x1 = np.meshgrid(xc, yc)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz0 = y_proba[:, 0].reshape(x0.shape)
zz1 = y_proba[:, 1].reshape(x0.shape)
zz2 = y_proba[:, 2].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour0 = plt.contour(x0, x1, zz0, cmap="hot",)
#contour1 = plt.contour(x0, x1, zz1, cmap="hot",)
#contour2 = plt.contour(x0, x1, zz2, cmap="hot",)

plt.clabel(contour0, inline =1)

plt.plot(X[y==2,0],X[y==2,1], 'g^', label='virginica')
plt.plot(X[y==1,0],X[y==1,1], 'bs', label='versicolor')
plt.plot(X[y==0,0],X[y==0,1], 'yo', label='setosa')

plt.ylabel('Petal width')
plt.xlabel('Petal length')
plt.axis([0,7,0,3.5])
plt.legend(loc='best')
plt.grid()
plt.show()
```
### Logistic Regression (from scratch)
```
for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_val @ Theta)
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    Theta = Theta - eta * gradients 
```
### SVM Classification plot
```
  w = svm_clf.coef_[0]
  b = svm_clf.intercept_[0]

  # At the decision boundary, w0*x0 + w1*x1 + b = 0
  # => x1 = -w0/w1 * x0 - b/w1
  x0 = np.linspace(xmin, xmax, 200)
  decision_boundary = -w[0] / w[1] * x0 - b / w[1]

  margin = 1/w[1]
  gutter_up = decision_boundary + margin
  gutter_down = decision_boundary - margin
  svs = svm_clf.support_vectors_

  xc = np.linspace(0,5.5, 500)
  yc= np.linspace(0, 3.5, 200)  
  x0, x1 = np.meshgrid(xc, yc)
  X_new = np.c_[x0.ravel(), x1.ravel()]  
  y_predict = svm_clf.predict(X_new)
  zz = y_predict.reshape(x0.shape)
  plt.contourf(x0, x1, zz, cmap='hot')
  zz0 = svm_clf.decision_function(X_new).reshape(x0.shape) #These measure the signed distance between each
  #instance and the decision boundary:

  contour0 = plt.contour(x0, x1, zz0, cmap="hot",levels=10)
  plt.clabel(contour0, inline =1)
```
### Incremental PCA 
```
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)

## OR ##

filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype='float32', mode='write',
shape=X_train.shape)
X_mmap[:] = X_train # could be a loop instead, saving the data
chunk by chunk
X_mmap.flush()

X_mmap = np.memmap(filename, dtype="float32",
mode="readonly").reshape(-1, 784)
batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)

```

