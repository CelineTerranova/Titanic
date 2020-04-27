

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

import os
for dirname, _, filenames in os.walk('/Users/celineterranova/Desktop/Data/Kaggle/Titanic'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/test.csv")
test_data.head()


# CHOOSE FEATURES AND VALUES FOR THE MODEL
y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
# features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
# train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

X = pd.get_dummies(train_data[features])

# THIS PART OF THE CODE IS SOLELY FOR MODEL FITTING PURPOSES
def get_mae(n, d, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
print("Percentage of errors on training data split: ")
print(mean_absolute_error(val_predictions, val_y))



# APPLY TO TEST DATA
# test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
y_test = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
