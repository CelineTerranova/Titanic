

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/train.csv")
test_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/test.csv")

# DEAL WITH TITLES
# Function that, given a title string, checks it and replaces it with the correct title
def title_corr(t):
    newt = t
    if t == 'Mrs' or t == 'Mr' or t == 'Miss':
        return newt
    elif t == 'Capt' or t == 'Col' or t == 'Major' or t == 'Dr' or t == 'Rev':
        newt = 'Crew'
    elif t == 'Jonkheer' or t == 'Sir' or t == 'the Countess' or t == 'Lady' or t == 'Master':
        newt = 'Noble'
    elif t == 'Don':
        newt = 'Mr'
    elif t == 'Dona' or t == 'Ms' or t == 'Mme':
        newt = 'Mrs'
    elif t == 'Mlle':
        newt = 'Miss'
    else: print("Title not included:", t)
    return newt

# Extract the titles from the name and put them in a list, then correct them
# Train data
train_data.insert(4,"Titles", "Empty")
titles = list()
for name in train_data["Name"]:
    titles.append(name.split(',')[1].split('.')[0].strip())
for i in range(len(titles)):
    titles[i] = title_corr(titles[i])
train_data["Titles"] = titles
# Test data
test_data.insert(3,"Titles", "Empty")
test_titles = list()
for name in test_data["Name"]:
    test_titles.append(name.split(',')[1].split('.')[0].strip())
for i in range(len(test_titles)):
    test_titles[i] = title_corr(test_titles[i])
test_data["Titles"] = test_titles


# FILL UP FARES THAT DON'T EXIST
train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())

# GROUP FARES
train_data.insert(7,"Fare Group", "Empty")
test_data.insert(7,"Fare Group", "Empty")
# See explore.py for the weighing of different fare groups
def group_fare(fare):
    if fare <= 170: return 0
    if fare > 170 and fare <= 340: return 1
    if fare > 340: return 2
# Loops over the df and fill the Fare Group column
for i, row in train_data.iterrows():
    train_data.at[i,'Fare Group'] = group_fare(row["Fare"])
# Same for test data
for i, row in test_data.iterrows():
    test_data.at[i,'Fare Group'] = group_fare(row["Fare"])


# FILL UP AGES THAT DON'T EXIST
# Function that returns the median age for passengers from a certain class, sex and title
def calc_age(df, cl, sx, tl):
    a = df.groupby(["Pclass", "Sex", "Titles"])["Age"].median()
    return a[cl][sx][tl]

# Merging datasets.
age_train = train_data.copy()
age_train.drop('PassengerId', axis=1, inplace=True)
age_train.drop('Survived',axis=1, inplace=True)
age_test = test_data.copy()
age_test.drop('PassengerId', axis=1, inplace=True)
df = pd.concat([age_train, age_test], sort=False).reset_index(drop=True)
# print(df.shape)

# Fill up missing ages
for i, row in train_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(df, row["Pclass"], row["Sex"], row["Titles"]))
        train_data.at[i,'Age'] = newage
    else: continue
# Same for test data
for i, row in test_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(df, row["Pclass"], row["Sex"], row["Titles"]))
        test_data.at[i,'Age'] = newage
    else: continue

# GROUP AGES
train_data.insert(5,"Age Group", "Empty")
test_data.insert(5,"Age Group", "Empty")
# See explore.py for the weighing of different age groups
def group_age(age):
    if age <= 16: return 4
    if age > 16 and age <= 32: return 1
    if age > 32 and age <= 48: return 2
    if age > 48 and age <= 64: return 3
    if age > 64: return 0

# Loops over the df and fill the Age Group column
for i, row in train_data.iterrows():
    train_data.at[i,'Age Group'] = group_age(row["Age"])
    # Same for test data
for i, row in test_data.iterrows():
    test_data.at[i,'Age Group'] = group_age(row["Age"])


# COMBINE SIBSP AND PARCH TO A SINGLE PARAMETER "FAMILY"
train_data.insert(9,"Family", "Empty")
train_data["Family"] = train_data["SibSp"] + train_data["Parch"]
test_data.insert(8,"Family", "Empty")
test_data["Family"] = test_data["SibSp"] + test_data["Parch"]

# FILL MISSING DATA FOR EMBARKED IN TRAINING DATA (NO MISSING IN TEST DATA)
train_data["Embarked"] = train_data["Embarked"].fillna('S')

# WEIGH EMBARKED WITH SURVIVAL RATE
def embarked_rate(embarked_port):
    if embarked_port == 'C': return 2
    if embarked_port == 'Q': return 1
    if embarked_port == 'S': return 0

train_data.insert(9,"Emb Rate", "Empty")
for i, row in train_data.iterrows():
    train_data.at[i,'Emb Rate'] = embarked_rate(row["Embarked"])
test_data.insert(9,"Emb Rate", "Empty")
for i, row in test_data.iterrows():
    test_data.at[i,'Emb Rate'] = embarked_rate(row["Embarked"])

# # DEAL WITH CATEGORICAL VARIABLES
# object_cols = ["Embarked", "Titles", "Sex"]
# from sklearn.preprocessing import OneHotEncoder
# # Apply one-hot encoder to each column with categorical data
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_data[object_cols]))
# OH_cols_test= pd.DataFrame(OH_encoder.transform(test_data[object_cols]))
# # One-hot encoding removed index; put it back
# OH_cols_train.index = train_data.index
# OH_cols_test.index = test_data.index
# # Remove categorical columns (will replace with one-hot encoding)
# num_train = train_data.drop(object_cols, axis=1)
# num_test = test_data.drop(object_cols, axis=1)
# # Add one-hot encoded columns to numerical features
# OH_train = pd.concat([num_train, OH_cols_train], axis=1)
# OH_test = pd.concat([num_test, OH_cols_test], axis=1)

# REPLACE SEX WITH NUMBERS
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)

# DROP USELESS COLUMNS
cols_to_drop = ["SibSp", "Parch", "Name", "Age", "Fare", "Cabin", "Ticket"]
# new_train = OH_train.drop(cols_to_drop, axis=1)
# new_test = OH_test.drop(cols_to_drop, axis=1)
new_train = train_data.drop(cols_to_drop, axis=1)
new_test = test_data.drop(cols_to_drop, axis=1)


# print(new_train.head())

# CHOOSE FEATURES AND VALUES FOR THE MODEL
y = new_train["Survived"]
# X = new_train.drop("Survived", axis=1)
# X_test = new_test
features = ["Pclass", "Sex", "Family", "Fare Group", "Titles", "Age Group"]



# # PREDICT ACCURACY
# from sklearn.model_selection import train_test_split
#
#
# predictors = pd.get_dummies(new_train[features])
# target = new_train["Survived"]
# x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#
# # MODEL, FIT AND PREDICT
# model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model1.fit(x_train, y_train)
# y1_pred = model1.predict(x_val)
#
# from xgboost import XGBClassifier
# model2 = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.05)
# model2.fit(x_train, y_train)
# y2_pred = model2.predict(x_val)
#
# from sklearn.svm import SVC
# model3 = SVC(random_state=1)
# model3.fit(x_train, y_train)
# y3_pred = model3.predict(x_val)
#
# from sklearn.ensemble import GradientBoostingClassifier
# model4 = GradientBoostingClassifier(random_state=42)
# model4.fit(x_train, y_train)
# y4_pred = model4.predict(x_val)
#
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import accuracy_score
# model1_acc = round(accuracy_score(y1_pred, y_val) * 100, 4)
# model2_acc = round(accuracy_score(y2_pred, y_val) * 100, 4)
# model3_acc = round(accuracy_score(y3_pred, y_val) * 100, 4)
# model4_acc = round(accuracy_score(y4_pred, y_val) * 100, 4)
#
# # print("Random Forest Accuracy:", model1_acc)
# # print("XGBoost Accuracy:", model2_acc)
# # print("SVC Accuracy:", model3_acc)
# # print("GB Accuracy:", model4_acc)
#
#
# models = pd.DataFrame({
#     'Model': ['Random Forest Classfier', 'XGBClassifier', 'SVC',
#               'Gradient Boosting Classifier'],
#     'Score': [model1_acc, model2_acc, model3_acc, model4_acc]})
# print(models.sort_values(by='Score', ascending=False))


X = pd.get_dummies(new_train[features])
X_test = pd.get_dummies(new_test[features])

# # MODEL, FIT AND PREDICT
model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model1.fit(X, y)
y1_test = model1.predict(X_test)

from xgboost import XGBClassifier
model2 = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.05)
model2.fit(X, y)
y2_test = model2.predict(X_test)

from sklearn.svm import SVC
model3 = SVC(random_state=1)
model3.fit(X,y)
y3_test = model3.predict(X_test)

from sklearn.ensemble import GradientBoostingClassifier
model4 = GradientBoostingClassifier(random_state=42)
model4.fit(X, y)
y4_test = model4.predict(X_test)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
model1_preds = cross_val_predict(model1, X, y, cv=10)
model1_acc = accuracy_score(y, model1_preds)
model2_preds = cross_val_predict(model2, X, y, cv=10)
model2_acc = accuracy_score(y, model2_preds)
model3_preds = cross_val_predict(model3, X, y, cv=10)
model3_acc = accuracy_score(y, model3_preds)
model4_preds = cross_val_predict(model4, X, y, cv=10)
model4_acc = accuracy_score(y, model4_preds)

print("Random Forest Accuracy:", model1_acc)
print("XGBoost Accuracy:", model2_acc)
print("SVC Accuracy:", model3_acc)
print("GB Accuracy:", model4_acc)


# HYPERPARAMETER TUNING
# Generating paramater grids for predictors
model1_param = [
    {'max_depth':[2,4,8],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10],
    'n_estimators':[100,300]}
]

model2_param = [
    {'max_depth':[2,4,8],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10],
    'learning_rate':[0.01, 0.05, 0.1],
    'n_estimators':[500,800]}
]

model3_param = [
    {'C':[0.001, 0.01, 0.1, 1.0, 10.0],
    'gamma':[0.001, 0.01, 0.1, 1.0],
    'kernel':['rbf', 'sigmoid']}
]

model4_param = [
    {'max_depth':[2,160],
    'min_samples_split':[2,4,6,10],
    'min_samples_leaf':[2,4,6,10],
    'learning_rate':[0.01, 0.05, 0.1],
    'n_estimators':[100,300],
    'subsample':[0.5, 1.0]}
]

from sklearn.model_selection import GridSearchCV
model1_grid = GridSearchCV(model1, model1_param, cv=5)
model1_grid.fit(X, y)
model1_best = model1_grid.best_estimator_
print("Model 1 done")
model2_grid = GridSearchCV(model2, model2_param, cv=5)
model2_grid.fit(X, y)
model2_best = model2_grid.best_estimator_
print("Model 2 done")
model3_grid = GridSearchCV(model3, model3_param, cv=5)
model3_grid.fit(X, y)
model3_best = model3_grid.best_estimator_
print("Model 3 done")
model4_grid = GridSearchCV(model4, model4_param, cv=5)
model4_grid.fit(X, y)
model4_best = model4_grid.best_estimator_
print("Model 4 done")


model1_best.fit(X, y)
model2_best.fit(X, y)
model3_best.fit(X, y)
model4_best.fit(X, y)

model1_best_preds = cross_val_predict(model1_best, X, y, cv=10)
model1_best_acc = accuracy_score(y, model1_best_preds)
model2_best_preds = cross_val_predict(model2_best, X, y, cv=10)
model2_best_acc = accuracy_score(y, model2_best_preds)
model3_best_preds = cross_val_predict(model3_best, X, y, cv=10)
model3_best_acc = accuracy_score(y, model3_best_preds)
model4_best_preds = cross_val_predict(model4_best, X, y, cv=10)
model4_best_acc = accuracy_score(y, model4_best_preds)

print("BEST Random Forest Accuracy:", model1_best_acc)
print("BEST XGBoost Accuracy:", model2_best_acc)
print("BEST SVC Accuracy:", model3_best_acc)
print("BEST GB Accuracy:", model4_best_acc)


model1_best_predict = model1_best.predict(X_test)
model2_best_predict = model2_best.predict(X_test)
model3_best_predict = model3_best.predict(X_test)
model4_best_predict = model4_best.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model1_best_predict})
output.to_csv('my_submission_Random.csv', index=False)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model2_best_predict})
output.to_csv('my_submission_XGBoost.csv', index=False)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model3_best_predict})
output.to_csv('my_submission_SVC.csv', index=False)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': model4_best_predict})
output.to_csv('my_submission_GB.csv', index=False)

print("Your submission was successfully saved!")
