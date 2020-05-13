#  THIS CODE IS USED TO EXPLORE DATA. THE GOAL IS TO FIND / ENGINEER THE BEST FEATURES
# WE ONLY WORK HERE WITH THE TRAINING DATA. ONCE THE BEST FEATURES ARE SELECTED, IT WILL BE APPLIED TO THE TEST DATA IN MYSUBMISSION.PY


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error





train_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/train.csv")
test_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/test.csv")

# print(train_data.describe())


import seaborn as sns
from matplotlib import pyplot as plt
import pylab as plot
params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'legend.loc': "best",

}
plot.rcParams.update(params)

train_data['Died'] = 1 - train_data['Survived']

# PLOT BY SEX (ABSOLUTE AND RELATIVE)
train_data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
train_data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# PLOT BY CLASS
train_data.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# PLOT BY NUMBER OF SPOUSES AND/OR SIBLINGS
train_data.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY NUMBER OF PARENTS AND/OR CHILDREN
train_data.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY FARE
plt.hist([train_data[train_data['Survived'] == 1]['Fare'], train_data[train_data['Survived'] == 0]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY FARE FOR EACH CLASS
plt.hist([train_data[train_data['Survived'] == 1][train_data['Pclass'] == 1]['Fare'], train_data[train_data['Survived'] == 0][train_data['Pclass'] == 1]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers (1st class)')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
plt.hist([train_data[train_data['Survived'] == 1][train_data['Pclass'] == 2]['Fare'], train_data[train_data['Survived'] == 0][train_data['Pclass'] == 2]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers (2nd class)')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
plt.hist([train_data[train_data['Survived'] == 1][train_data['Pclass'] == 3]['Fare'], train_data[train_data['Survived'] == 0][train_data['Pclass'] == 3]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers (3rd class)')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# PLOT BY TITLE
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

# Extract the titles from the name and put them in a list
titles = list()
for name in train_data["Name"]:
    titles.append(name.split(',')[1].split('.')[0].strip())

for i in range(len(titles)):
    titles[i] = title_corr(titles[i])

train_data["Titles"] = titles

plt.hist([train_data[train_data['Survived'] == 1]['Titles'], train_data[train_data['Survived'] == 0]['Titles']], label = ['Survived','Dead'])
plt.xlabel('Title')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY AGE
plt.hist([train_data[train_data['Survived'] == 1]['Age'], train_data[train_data['Survived'] == 0]['Age']], bins = 8, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY AGE FOR MEN AND WOMEN
plt.hist([train_data[train_data['Survived'] == 1][train_data['Sex'] == 'female']['Age'], train_data[train_data['Survived'] == 0][train_data['Sex'] == 'female']['Age']], bins = 8, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of women')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
plt.hist([train_data[train_data['Survived'] == 1][train_data['Sex'] == 'male']['Age'], train_data[train_data['Survived'] == 0][train_data['Sex'] == 'male']['Age']], bins = 8, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of men')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# DIVIDE AGE IN GROUPS AND CALCULATE SURVIVAL RATE FOR EACH
train_data['AgeGroup'] = pd.cut(train_data['Age'],5)
print(train_data[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False))

# DIVIDE FARES IN GROUPS AND CALCULATE SURVIVAL RATE FOR EACH
train_data['AgeGroup'] = pd.cut(train_data['Fare'],3)
print(train_data[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False))

# PLOT BY EMBARKED
train_data["Embarked"] = train_data["Embarked"].fillna('S')
plt.hist([train_data[train_data['Survived'] == 1]['Embarked'], train_data[train_data['Survived'] == 0]['Embarked']], label = ['Survived','Dead'])
plt.xlabel('Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)')
plt.ylabel('Number of passengers')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# # DIVIDE EMBARKED IN GROUPS AND CALCULATE SURVIVAL RATE FOR EACH
print(train_data[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values('Survived', ascending=False))
