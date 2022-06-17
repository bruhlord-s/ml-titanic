import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = pd.read_csv('datasets/train.csv')
X_test = pd.read_csv('datasets/test.csv')

# Drop rows w/o survived value
# set y and drop survived column from x
X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X['Survived']
X.drop(['Survived'], axis=1, inplace=True)

useless_cols = ['Name', 'Ticket', 'Embarked', 'Cabin', 'SibSp']
X.drop(useless_cols, axis=1, inplace=True)
X_test.drop(useless_cols, axis=1, inplace=True)

# Drop columns with nan values
# cols_with_missing = [col for col in X.columns
#                      if X[col].isnull().any()]
# X.drop(cols_with_missing, axis=1, inplace=True)
# X_test.drop(cols_with_missing, axis=1, inplace=True)

s = (X.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

OH_cols_train.index = X.index
OH_cols_test.index = X_test.index

num_X = X.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

OH_X = pd.concat([num_X, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(OH_X))
imputed_X_test = pd.DataFrame(my_imputer.transform(OH_X_test))

imputed_X_train.columns = OH_X.columns
imputed_X_test.columns = OH_X_test.columns

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(imputed_X_train, y)
predicts = model.predict(imputed_X_test)

result = pd.DataFrame().assign(PassengerId=X_test['PassengerId'].astype('int'), Survived=np.rint(predicts).astype('int'))
result.to_csv('datasets/result.csv', index=False)
