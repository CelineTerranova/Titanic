

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# Function that returns a table with the median age for passengers from a certain class, sex and title
def calc_age(df, cl, sx, tl):
    a = df.groupby(["Pclass", "Sex", "Titles"])["Age"].median()
    return a[cl][sx][tl]

# Loops over the df and replace the missing ages (train)
for i, row in train_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(train_data, row["Pclass"], row["Sex"], row["Titles"]))
        train_data.at[i,'Age'] = newage
    else: continue
# Same for test data
for i, row in test_data.iterrows():
    if pd.isna(row['Age']) :
        newage = (calc_age(test_data, row["Pclass"], row["Sex"], row["Titles"]))
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

# # DEAL WITH CATEGORICAL VARIABLES
# cat_cols = ["Embarked"]
# from sklearn.preprocessing import OneHotEncoder
# OHE = OneHotEncoder()
# df["OHE Embarked"] = clf.transform(np.array(df["Name"]).reshape(-1, 1))
#
# # Apply one-hot encoder to each column with categorical data
# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# OH_cols_train = train_data.DataFrame(OH_encoder.fit_transform(train_data[object_cols]))
# OH_cols_test = test_data.DataFrame(OH_encoder.transform(test_data[object_cols]))
# # One-hot encoding removed index; put it back
# OH_cols_train.index = train_data.index
# OH_cols_test.index = test_data.index


# DROP USELESS COLUMNS
cols_to_drop = ["SibSp", "Parch", "Name", "Age", "Fare"]
new_train = train_data.drop(cols_to_drop, axis=1)
new_test = test_data.drop(cols_to_drop, axis=1)

# CHOOSE FEATURES AND VALUES FOR THE MODEL
y = new_train["Survived"]
features = ["Pclass", "Sex", "Family", "Fare Group", "Titles", "Age Group", "Embarked"]
X = pd.get_dummies(new_train[features])
X_test = pd.get_dummies(new_test[features])

# MODEL, FIT AND PREDICT
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
y_test = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
