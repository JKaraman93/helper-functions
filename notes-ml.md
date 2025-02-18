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
