##Kaggle Learning from Titanic Disaster

## Help from:
#Ensembling:
## https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Feature Manipulation
## https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# General Info and feature Manipulation
## https://www.kaggle.com/startupsci/titanic-data-science-solutions

## Much of the code was coppied directly from the above and does not represnet my work.

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

class Data(object):
	def __init__(self, training="train.csv",testing="test.csv"):
		self.train = pd.read_csv(training)
		self.test = pd.read_csv(testing)
		self.full_data = [self.train, self.test]
		self.drop_elements = ['Title', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']

	#Create Family Size Feature
	def Family(self):
		for dataset in self.full_data:
			dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+ 1

	# Create Feature "traveling alone"
	def Traveling_Alone(self):
		for dataset in self.full_data:
			dataset['IsAlone'] = 0
			dataset.loc[dataset['FamilySize'] == 1, 'IsAlone']=1

	# Replace unmarked Values in embarked with a new value, x === not what is done in the tutorial
	def Embarked(self):
		for dataset in self.full_data:
			dataset['Embarked'] = dataset['Embarked'].fillna('X')

		# # Mapping Embarked
		for dataset in self.full_data:
			dataset['EmbS'] = 0
			dataset['EmbC'] = 0
			dataset['EmbQ'] = 0
			dataset['EmbX'] = 0
			dataset.loc[dataset['Embarked'] == 'S', 'EmbS']=1
			dataset.loc[dataset['Embarked'] == 'C', 'EmbC']=1
			dataset.loc[dataset['Embarked'] == 'Q', 'EmbQ']=1
			dataset.loc[dataset['Embarked'] == 'X', 'EmbX']=1


	# Divide Fare into four catagories.
	def Divide_Fare(self):
		for dataset in self.full_data:
			dataset['Fare'] = dataset['Fare'].fillna(self.train['Fare'].median())
			self.train['CategoricalFare'] = pd.qcut(self.train['Fare'], 4)

		for dataset in self.full_data:
		 	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
		 	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
		 	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
		 	dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
		 	dataset['Fare'] = dataset['Fare'].astype(int)


	#Filling Age Data
	#I dont know how i feel about adding random nubmers into the data like the tutorial did.  I understand if it's just a little data:
	#But then again, when I rand this a few times, it didn't change the output all that much -- so i'll just go with it.
	def Age(self):
		for dataset in self.full_data:
		    age_avg 	   = dataset['Age'].mean()
		    age_std 	   = dataset['Age'].std()
		    age_null_count = dataset['Age'].isnull().sum()
		    
		    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
		    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
		    dataset['Age'] = dataset['Age'].astype(int)
		    
		self.train['CategoricalAge'] = pd.cut(self.train['Age'], 5)

		for dataset in self.full_data:
			dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
			dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
			dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
			dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
			dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

	#Get titles:
	def Titles(self):
		def get_title(name):
			title_search = re.search(' ([A-Za-z]+)\.', name)
		# 	If the title exists, extract and return it.
			if title_search:
 				return title_search.group(1)
 			return ""

		for dataset in self.full_data:
			dataset['Title'] = dataset['Name'].apply(get_title)

		for dataset in self.full_data:
			dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 			'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


    	# Mapping titles - tutoral kept this as 1-5
		for dataset in self.full_data:
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

	#Make sex binary
	def Sex(self):
		for dataset in self.full_data:
			dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Feature selection
 	def Drop_features(self):
		self.train = self.train.drop(self.drop_elements, axis = 1)
		self.train = self.train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
		self.test  = self.test.drop(self.drop_elements, axis = 1)


	### Clean all data
	def clean(self):
		self.Family()
		self.Traveling_Alone()
		self.Embarked()
		self.Divide_Fare()
		self.Age()
		self.Titles()
		self.Sex()
		self.Drop_features()

class Train_Model(Data):

	def __init__(self, training="train.csv",testing="test.csv"):
		Data.__init__(self, training="train.csv",testing="test.csv")
		Data.clean(self)
		self.X_train = self.train.drop("Survived", axis = 1)
		self.Y_train = self.train["Survived"]
		self.X_test = self.test

	def visual(self):
		colormap = plt.cm.viridis
		plt.figure(figsize=(11,9))
		plt.title('Pearson Correlation of Features', y=1.05, size=15)
		sns.heatmap(self.train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
		plt.show()

	def test_models(self):

		logreg = LogisticRegression()
		logreg.fit(self.X_train, self.Y_train)
		Y_pred = logreg.predict(self.X_test)
		acc_log = round(logreg.score(self.X_train, self.Y_train) * 100, 2)
		acc_log

		svc = SVC()
		svc.fit(self.X_train, self.Y_train)
		Y_pred = svc.predict(self.X_test)
		acc_svc = round(svc.score(self.X_train, self.Y_train) * 100, 2)
		acc_svc

		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(self.X_train, self.Y_train)
		Y_pred = knn.predict(self.X_test)
		acc_knn = round(knn.score(self.X_train, self.Y_train) * 100, 2)
		acc_knn

		gaussian = GaussianNB()
		gaussian.fit(self.X_train, self.Y_train)
		Y_pred = gaussian.predict(self.X_test)
		acc_gaussian = round(gaussian.score(self.X_train, self.Y_train) * 100, 2)
		acc_gaussian

		perceptron = Perceptron()
		perceptron.fit(self.X_train, self.Y_train)
		Y_pred = perceptron.predict(self.X_test)
		acc_perceptron = round(perceptron.score(self.X_train, self.Y_train) * 100, 2)
		acc_perceptron

		linear_svc = LinearSVC()
		linear_svc.fit(self.X_train, self.Y_train)
		Y_pred = linear_svc.predict(self.X_test)
		acc_linear_svc = round(linear_svc.score(self.X_train, self.Y_train) * 100, 2)
		acc_linear_svc

		sgd = SGDClassifier()
		sgd.fit(self.X_train, self.Y_train)
		Y_pred = sgd.predict(self.X_test)
		acc_sgd = round(sgd.score(self.X_train, self.Y_train) * 100, 2)
		acc_sgd

		decision_tree = DecisionTreeClassifier()
		decision_tree.fit(self.X_train, self.Y_train)
		Y_pred = decision_tree.predict(self.X_test)
		acc_decision_tree = round(decision_tree.score(self.X_train, self.Y_train) * 100, 2)
		acc_decision_tree

		random_forest = RandomForestClassifier(n_estimators=100)
		random_forest.fit(self.X_train, self.Y_train)
		Y_pred = random_forest.predict(self.X_test)
		random_forest.score(self.X_train, self.Y_train)
		acc_random_forest = round(random_forest.score(self.X_train, self.Y_train) * 100, 2)
		acc_random_forest

		models = pd.DataFrame({
    		'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    		'Score': [acc_svc, acc_knn, acc_log, 
           	   acc_random_forest, acc_gaussian, acc_perceptron, 
           	   acc_sgd, acc_linear_svc, acc_decision_tree]})
		return models.sort_values(by='Score', ascending=False)

### The following is coppied pretty exactly from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer