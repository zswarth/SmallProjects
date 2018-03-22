## Help from:
###### Ensembling:
## https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
###### Feature Manipulation
## https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
### General Info and feature Manipulation
######## https://www.kaggle.com/startupsci/titanic-data-science-solutions


import pandas as pd
import numpy as np
import re

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full_data = [train, test]

### Create Family Size feature
for dataset in full_data:
	dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+ 1

### Create a paramter 'traveling alone"'
for dataset in full_data:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone']=1

### Replace unmarked Values in embarked with a new value, x === not what is done in the tutorial
for dataset in full_data:
	dataset['Embarked'] = dataset['Embarked'].fillna('X')


### Divide Fare into four catagories.
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

## I dont know how i feel about adding random nubmers into the data like the tutorial did.  I understand if it's just a little data:
## But this seems to be missing quite a bit
#### But then again, when I rand this a few times, it didn't change the output all that much -- so i'll just go with it.
for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)



### Get titles:
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


### Cleaning data:
###### The tutorial puts non binary things into numerical digits.  I think it might be better to have several binary colums --- especially under embarked and titels


for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


    # Mapping titles
for dataset in full_data:
	dataset['Mr'] = 0
	dataset['Miss'] = 0
	dataset['Mrs'] = 0
	dataset['Master'] = 0
	dataset['Rare'] = 0
	dataset.loc[dataset['Title'] == 'Mr', 'Mr']=1
	dataset.loc[dataset['Title'] == 'Miss', 'Miss']=1
	dataset.loc[dataset['Title'] == 'Mrs', 'Mrs']=1
	dataset.loc[dataset['Title'] == 'Master', 'Master']=1
	dataset.loc[dataset['Title'] == 'Rare', 'Rare']=1

# # Mapping Embarked
for dataset in full_data:
	dataset['EmbS'] = 0
	dataset['EmbC'] = 0
	dataset['EmbQ'] = 0
	dataset['EmbX'] = 0
	dataset.loc[dataset['Embarked'] == 'S', 'EmbS']=1
	dataset.loc[dataset['Embarked'] == 'C', 'EmbC']=1
	dataset.loc[dataset['Embarked'] == 'Q', 'EmbQ']=1
	dataset.loc[dataset['Embarked'] == 'X', 'EmbX']=1

    
 #Mapping Fare
 ### Keeping this as it is in the tutorial.  Going to see if it does better or worse if I cahnge it later like the previous two
 	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
 	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
 	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
 	dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
 	dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Feature Selection
drop_elements = ['Title', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)


################ Train Model ###############

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier




X_train = train.drop("Survived", axis = 1)
Y_train = train["Survived"]
X_test = test

