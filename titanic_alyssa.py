import pandas as pd
import numpy as np
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class Data(object):
	def __init__(self, training="train.csv",testing="test.csv", train_percentage = .8):
		self.train = pd.read_csv(training)
		self.test = pd.read_csv(testing)
		self.PassengerId = self.test['PassengerId']
		self.full_data = [self.train, self.test]
		self.drop_elements = ['Title', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'Cabin', 'SibSp','Parch', 'FamilySize']
		self.clean()

		self.Male = self.train[self.train['Sex']==1]
		self.Female = self.train[self.train['Sex'] == 0]
		self.Male_target = self.Male["Survived"]
		self.Female_target = self.Female["Survived"]
		self.Male_train = self.Male.drop("Survived", axis = 1) 
		self.Female_train = self.Female.drop("Survived", axis = 1)


		self.Testing = self.test
		self.Testing['PassengerId'] = self.PassengerId   
		self.Testing_Male = self.Testing[self.Testing['Sex']==1]
		self.Testing_Female = self.Testing[self.Testing['Sex']==0]
		self.Testing_ID_Male = self.Testing_Male['PassengerId']
		self.Testing_Male = self.Testing_Male.drop('PassengerId', axis = 1)
		self.Testing_ID_Female = self.Testing_Female['PassengerId']
		self.Testing_Female = self.Testing_Female.drop('PassengerId', axis = 1)

		# self.X_train = self.train.drop("Survived", axis = 1)
		# self.Y_train = self.train["Survived"]


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


class Train(Data):
	def __init__(self, training="train.csv",testing="test.csv"):
		Data.__init__(self, training="train.csv",testing="test.csv")
		self.clf_male = RandomForestClassifier(n_estimators=100, max_leaf_nodes=12, max_depth=12).fit(self.Male_train, self.Male_target)
		self.clf_female = RandomForestClassifier(n_estimators=100, max_leaf_nodes=12, max_depth=12).fit(self.Female_train, self.Female_target)

		self.Male_results = self.clf_male.predict(self.Testing_Male)
		self.Female_results = self.clf_female.predict(self.Testing_Female)

		self.Submission1 = pd.DataFrame({ 'PassengerId': self.Testing_ID_Male,'Survived': self.Male_results})
		self.Submission2 = pd.DataFrame({ 'PassengerId': self.Testing_ID_Female,'Survived': self.Female_results})
		self.Submission = pd.concat([self.Submission1, self.Submission2])
