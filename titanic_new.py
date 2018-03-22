import pandas as pd
import numpy as np
import re



from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import seaborn as sns
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

class Data(object):
	def __init__(self, training="train.csv",testing="test.csv", train_percentage = .8):
		self.train = pd.read_csv(training)
		self.test = pd.read_csv(testing)
		self.Y_train = self.train["Survived"]
		self.PassengerId = self.test['PassengerId']
		self.full_data = [self.train, self.test]
		self.Titles()
		self.Age()
		self.Cabin()
		self.full_data = [self.train, self.test]
		self.drop_elements = ['Title', 'Embarked', 'Age', 'PassengerId', 'Name', 'Ticket', 'SibSp','Parch', 'FamilySize']
		self.clean()
		self.X_train = self.train #this is a holdover from an older thing -- didn't feel like rewriting the variables below; will change later
		self.X_train_cv, self.X_test_cv, self.y_train_cv, self.y_test_cv = self.cross_validation_sets(pt = train_percentage)


	def cross_validation_sets(self, pt = .8):
		from sklearn.model_selection import train_test_split
		return train_test_split(self.X_train, self.Y_train, test_size=(1 - pt), train_size = pt)

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

	def Cabin(self):
		# got this from ahmedbesbes.com --- i like this for making catgorical binary collums (this called one hot array?)
		self.combine.Cabin.fillna('None', inplace = True)
		self.combine['Cabin'] = self.combine['Cabin'].map(lambda c:c[0])
		cabin_dummies = pd.get_dummies(self.combine['Cabin'],prefix='Cabin')
		self.combine = pd.concat([self.combine, cabin_dummies], axis=1)
		self.combine.drop('Cabin', axis=1, inplace = True)

		self.train = self.combine.head(891)
		self.test = self.combine.iloc[891:]


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
	# I'm taking most of this code (altered slightly from http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
	def Age(self):
		#Just cause i'm using their code, instead of my own, chagning some of the ways i'm manipulating the data
		self.combine = self.train.append(self.test)
		self.combine.drop('Survived', 1, inplace = True)
		self.combine.reset_index(inplace = True)
		self.combine.drop('index', inplace=True, axis=1)

		gtrain = self.combine.head(891).groupby(['Sex','Pclass','Title'])
		gtrain_median = gtrain.median()
		gtest = self.combine.iloc[891:].groupby(['Sex','Pclass','Title'])
		gtest_median = gtest.median()

		def fillAges(row, grouped_median):
			if row['Sex']=='female' and row['Pclass'] == 1:
				if row['Title'] == 'Miss':
					return grouped_median.loc['female', 1, 'Miss']['Age']
				elif row['Title'] == 'Mrs':
					return grouped_median.loc['female', 1, 'Mrs']['Age']
				elif row['Title'] == 'Mr':
					return grouped_median.loc['female', 1, 'Mr']['Age']
				elif row['Title'] == 'Officer':
					return grouped_median.loc['female', 1, 'Officer']['Age']
				elif row['Title'] == 'Royalty':
					return grouped_median.loc['female', 1, 'Royalty']['Age']

			elif row['Sex']=='female' and row['Pclass'] == 2:
				if row['Title'] == 'Miss':
					return grouped_median.loc['female', 2, 'Miss']['Age']
				elif row['Title'] == 'Mrs':
					return grouped_median.loc['female', 2, 'Mrs']['Age']


			elif row['Sex']=='female' and row['Pclass'] == 3:
				if row['Title'] == 'Miss':
					return grouped_median.loc['female', 3, 'Miss']['Age']
				elif row['Title'] == 'Mrs':
					return grouped_median.loc['female', 3, 'Mrs']['Age']

			elif row['Sex']=='male' and row['Pclass'] == 1:
				if row['Title'] == 'Master':
					return grouped_median.loc['male', 1, 'Master']['Age']
				elif row['Title'] == 'Mr':
					return grouped_median.loc['male', 1, 'Mr']['Age']
				elif row['Title'] == 'Officer':
					return grouped_median.loc['male', 1, 'Rare']['Age']
				elif row['Title'] == 'Royalty':
					return grouped_median.loc['male',1,'Royalty']['Age']

			elif row['Sex']=='male' and row['Pclass'] == 2:
				if row['Title'] == 'Master':
					return grouped_median.loc['male', 2, 'Master']['Age']
				elif row['Title'] == 'Mr':
					return grouped_median.loc['male', 2, 'Mr']['Age']
				elif row['Title'] == 'Officer':
					return grouped_median.loc['male', 2, 'Officer']['Age']

			elif row['Sex']=='male' and row['Pclass'] == 3:
				if row['Title'] == 'Master':
					return grouped_median.loc['male', 3, 'Master']['Age']
				elif row['Title'] == 'Mr':
					return grouped_median.loc['male', 3, 'Mr']['Age']
				elif row['Title'] == 'Officer':
					return grouped_median.loc['male', 3, 'Officer']['Age']
	    
		self.combine.head(891).Age = self.combine.head(891).apply(lambda r : fillAges(r, gtrain_median) if np.isnan(r['Age']) 
	                                                      else r['Age'], axis=1)
	    
		self.combine.iloc[891:].Age = self.combine.iloc[891:].apply(lambda r : fillAges(r, gtest_median) if np.isnan(r['Age']) 
	                                                      else r['Age'], axis=1)
		   

		self.train = self.combine.head(891)
		self.test = self.combine.iloc[891:]


		# for dataset in self.full_data:
		#     age_avg 	   = dataset['Age'].mean()
		#     age_std 	   = dataset['Age'].std()
		#     age_null_count = dataset['Age'].isnull().sum()
		    
		#     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
		#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
		#     dataset['Age'] = dataset['Age'].astype(int)
		    
		# self.train['CategoricalAge'] = pd.cut(self.train['Age'], 5)
		self.full_data = [self.train, self.test]
		for dataset in self.full_data:
			dataset['Age1'] = 0
			dataset['Age2'] = 0
			dataset['Age3'] = 0
			dataset['Age4'] = 0
			dataset['Age5'] = 0
			
			
			dataset.loc[ dataset['Age'] <= 16, 'Age1'] = 1
			dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age2'] = 1
			dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age3'] = 1
			dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age4'] = 1
			dataset.loc[ dataset['Age'] > 64, 'Age5']                           = 1

	#Get titles:
	def Titles(self):
		#lets try using Ahmed's titles
		def get_title(name):
					title_search = re.search(' ([A-Za-z]+)\.', name)
				# 	If the title exists, extract and return it.
					if title_search:
						return title_search.group(1)
					return ""

				for dataset in self.full_data:
					dataset['Title'] = dataset['Name'].apply(get_title)

				for dataset in self.full_data:
					dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',\
		 			'Dr', 'Major', 'Rev'], 'Officer')
					dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
					dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
					dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
					dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess',\
		 			'Don', 'Sir', 'Jonkheer', 'Dona'], 'Royalty')

		# def get_title(name):
		# 	title_search = re.search(' ([A-Za-z]+)\.', name)
		# # 	If the title exists, extract and return it.
		# 	if title_search:
 	# 			return title_search.group(1)
 	# 		return ""

		# for dataset in self.full_data:
		# 	dataset['Title'] = dataset['Name'].apply(get_title)

		# for dataset in self.full_data:
		# 	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	# 		'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
  #   		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  #   		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
  #   		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


    	# Mapping titles - tutoral kept this as 1-5
		for dataset in self.full_data:
			dataset['Mr'] = 0
			dataset['Miss'] = 0
			dataset['Mrs'] = 0
			dataset['Master'] = 0
			dataset['Royalty'] = 0
			dataset['Officer'] = 0
			dataset.loc[dataset['Title'] == 'Mr', 'Mr']=1
			dataset.loc[dataset['Title'] == 'Miss', 'Miss']=1
			dataset.loc[dataset['Title'] == 'Mrs', 'Mrs']=1
			dataset.loc[dataset['Title'] == 'Master', 'Master']=1
			dataset.loc[dataset['Title'] == 'Royalty', 'Royalty']=1
			dataset.loc[dataset['Title'] == 'Officer', 'Officer']=1
	#Make sex binary
	def Sex(self):
		for dataset in self.full_data:
			dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Feature selection
 	def Drop_features(self):
		self.train = self.train.drop(self.drop_elements, axis = 1)
		self.train = self.train.drop(['CategoricalFare'], axis = 1)
		self.test  = self.test.drop(self.drop_elements, axis = 1)

	### Clean all data
	def clean(self):
		self.Family()
		self.Traveling_Alone()
		self.Embarked()
		self.Divide_Fare()
		self.Sex()
		self.Drop_features()

class Train_Model(Data):

	def __init__(self, training="train.csv",testing="test.csv", train_percentage=.8, classifier = RandomForestClassifier(n_estimators = 25)):
		Data.__init__(self, training="train.csv",testing="test.csv", train_percentage= train_percentage)
		self.clf = classifier
		self.training()
		self.predict()

	def visual(self):
		colormap = plt.cm.viridis
		plt.figure(figsize=(11,9))
		plt.title('Pearson Correlation of Features', y=1.05, size=15)
		sns.heatmap(self.train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
		plt.show()

	def training(self, cv = False):
		self.clf.fit(self.X_train_cv, self.y_train_cv)
		if cv:
			print cross_val_score(self.clf, self.X_train, self.Y_train)

	def reduce_features(self):
		#from ahmedbesbes.com
		self.training()
		self.features = pd.DataFrame()
		self.features['feature']=self.train.columns
		self.features['importance'] = self.clf.feature_importances_
		self.features.sort_values(by=['importance'], ascending=True, inplace=True)
		self.features.set_index('feature', inplace=True)


		self.new_model = SelectFromModel(self.clf, prefit=True)
		self.train_reduced = self.new_model.transform(self.train)
		self.test_reduced = self.new_model.transform(self.test)

		## add in tuning peramters from ahmedbesbes.com
		## Dont completly understand this -- go through the literature on Rando Forests.

		# run_gs = False

		# if run_gs:
		# 	parameter_grid = {
		# 		'max_depth' : [4, 6, 8],
		# 		'n_estimators': [50, 10],
		# 		'max_features': ['sqrt', 'auto', 'log2'],
		# 		'min_samples_split': [1, 3, 10],
		#   		'min_samples_leaf': [1, 3, 10],
		# 		'bootstrap': [True, False],
		#       		}
		# 	forest = RandomForestClassifier()
		# 	cross_validation = StratifiedKFold(targets, n_folds=5)

		# 	grid_search = GridSearchCV(forest,
		#                                scoring='accuracy',
		#                                param_grid=parameter_grid,
		#                                cv=cross_validation)

		# 	grid_search.fit(self.X_train, self.Y_train)
		# 	model = grid_search
		# 	parameters = grid_search.best_params_

		# 	print('Best score: {}'.format(grid_search.best_score_))
		# 	print('Best parameters: {}'.format(grid_search.best_params_))
		# else: 
		# 	parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
		# 		'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
		    
		# 	self.model = RandomForestClassifier(**parameters)
		# 	self.model.fit(self.X_train, self.Y_train)

	def compute_score(clf, X, y, scoring='accuracy'):
		xval = cross_val_score(clf, X, y, cv = 5, scoring='accuracy')
		return np.mean(xval)


	def combine_models(self):
		self.clf_SVC = SVC(kernel = 'linear', C = 1).fit(self.X_train, self.Y_train)
		self.clf_LR = LogisticRegression().fit(self.X_train, self.Y_train)
		self.clf_RF = RandomForestClassifier(n_estimators=100, max_leaf_nodes=12, max_depth=12).fit(self.X_train, self.Y_train)
		self.clf_RF2 = RandomForestClassifier(n_estimators = 40, max_depth = 4, max_features = 'auto', min_samples_leaf = 1, bootstrap = True).fit(self.X_train, self.Y_train)
		self.clf_KNN = KNeighborsClassifier(n_neighbors = 6).fit(self.X_train, self.Y_train)
		self.clf_GNB = GaussianNB().fit(self.X_train, self.Y_train)
		self.clf_PCT = Perceptron().fit(self.X_train, self.Y_train)
		self.clf_SGD = SGDClassifier().fit(self.X_train, self.Y_train)
		self.clf_DT = DecisionTreeClassifier().fit(self.X_train, self.Y_train)
		self.clf_XGB = XGBClassifier().fit(self.X_train, self.Y_train)
		self.clf_MPL = MLPClassifier(hidden_layer_sizes=(100,4)).fit(self.X_train, self.Y_train)

		self.predictSVC = self.clf_SVC.predict(self.test)
		self.predictLR = self.clf_LR.predict(self.test)
		self.predictRF = self.clf_RF.predict(self.test)
		self.predictRF2 = self.clf_RF2.predict(self.test)
		self.predictKNN = self.clf_KNN.predict(self.test)
		self.predictGNB = self.clf_GNB.predict(self.test)
		# self.predictPCT = self.clf_PCT.predict(self.test)
		# self.predictSGD = self.clf_SGD.predict(self.test)
		# self.predictDT = self.clf_DT.predict(self.test)
		self.predictXGB = self.clf_XGB.predict(self.test)
		self.predictMPL = self.clf_MPL.predict(self.test)
		self.combine = pd.DataFrame({ 'MPL':self.predictMPL, 'SVC': self.predictSVC,'LR': self.predictLR,'RF': self.predictRF,'RF': self.predictRF2,'KNN': self.predictKNN,'GNB': self.predictGNB, 'XBG': self.predictXGB})
		self.combine['mean'] = self.combine.mean(axis=1)
		self.combine['mean2'] = self.combine['mean'].round(0).astype(int)
		# self.combine['actual'] = self.y_test_cv.values


	def predict(self):
		self.predicted = self.clf.predict(self.X_test_cv)
		self.Results = self.X_test_cv
		self.Results['Actually Survived'] = self.y_test_cv
		self.Results['Predicted Survived'] = self.predicted

		self.Results['Correct'] = np.where(self.Results['Actually Survived']==self.Results['Predicted Survived'], True, False)

	def export_result(self):
		## Train on Full Data Set:
		self.clf.fit(self.X_train, self.Y_train)
		results = self.clf.predict(self.test)


		Submission = pd.DataFrame({ 'PassengerId': self.PassengerId,'Survived': results})
		Submission.to_csv("Submission2.csv", index=False)

class Visualize_Model(Train_Model):
	import matplotlib.pyplot as plt
	import seaborn as sns
	def __init__(self, training="train.csv",testing="test.csv", train_percentage=.8, classifier = RandomForestClassifier(n_estimators = 25)):
		Train_Model.__init__(self, training="train.csv",testing="test.csv", train_percentage=.8, classifier = RandomForestClassifier(n_estimators = 25))

	def feature_importance(self):
		importance = self.clf.feature_importances_
		plt.plot(importance)
		plt.show()

	def CorrectBySex(self):
		correct = self.Results[self.Results['Correct']==True]['Sex'].value_counts()
		incorrect = self.Results[self.Results['Correct']==False]['Sex'].value_counts()
		df = pd.DataFrame([correct,incorrect])
		df.index = ['Correct','Incorrect']
		df.plot(kind='bar',stacked=True, figsize=(15,8))
		plt.show()

	def CorrectByClass(self):
		correct = self.Results[self.Results['Correct']==True]['Pclass'].value_counts()
		incorrect = self.Results[self.Results['Correct']==False]['Pclass'].value_counts()
		df = pd.DataFrame([correct,incorrect])
		df.index = ['Correct','Incorrect']
		df.plot(kind='bar',stacked=True, figsize=(15,8))
		plt.show()


	def CorrectByClass(self):
		correct = self.Results[self.Results['Correct']==True]['Fare'].value_counts()
		incorrect = self.Results[self.Results['Correct']==False]['Fare'].value_counts()
		df = pd.DataFrame([correct,incorrect])
		df.index = ['Correct','Incorrect']
		df.plot(kind='bar',stacked=True, figsize=(15,8))
		plt.show()

	def CorrectByAlone(self):
		correct = self.Results[self.Results['Correct']==True]['IsAlone'].value_counts()
		incorrect = self.Results[self.Results['Correct']==False]['IsAlone'].value_counts()
		df = pd.DataFrame([correct,incorrect])
		df.index = ['Correct','Incorrect']
		df.plot(kind='bar',stacked=True, figsize=(15,8))
		plt.show()