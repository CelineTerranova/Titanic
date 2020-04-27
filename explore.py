#  THIS CODE IS USED TO EXPLORE DATA. THE GOAL IS TO FIND / ENGINEER THE BEST FEATURES
# WE ONLY WORK HERE WITH THE TRAINING DATA. ONCE THE BEST FEATURES ARE SELECTED, IT WILL BE APPLIED TO THE TEST DATA IN MYSUBMISSION.PY


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error





train_data = pd.read_csv("/Users/celineterranova/Desktop/Data/Kaggle/Titanic/train.csv")
# print(train_data.shape)
# print(train_data.head())
print(train_data.describe())
#train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())


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
# train_data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', stacked=True)
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# train_data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.show()

# PLOT BY AGE
# fig = plt.figure(figsize=(25, 7))
# sns.violinplot(x='Sex', y='Age',
#                hue='Survived', data=train_data,
#                split=True,
#                #palette={0: "r", 1: "g"}
#               );
# plt.show()

# PLOT BY SEX AND AGE
# train_data['Died'] = 1 - train_data['Survived']
# train_data.groupby('Age').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.show()


# PLOT BY CLASS
# train_data.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
# plt.show()


# PLOT BY NUMBER OF SPOUSES AND/OR SIBLINGS
train_data.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()

# PLOT BY NUMBER OF PARENS AND/OR CHILDREN
train_data.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
