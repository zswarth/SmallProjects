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

import plotly.offline as py
py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.tools as tls

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
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;


class Data(object):
	def __init__(self, training="train.csv",testing="test.csv", train_percentage = .8):
		self.train = pd.read_csv(training)
		self.test = pd.read_csv(testing)
		self.PassengerId = self.test['PassengerId']
		self.full_data = [self.train, self.test]
		self.drop_elements = ['Title', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
		self.clean()
		self.X_train = self.train.drop("Survived", axis = 1)
		self.Y_train = self.train["Survived"]
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

	def __init__(self, training="train.csv",testing="test.csv", train_percentage=.8):
		Data.__init__(self, training="train.csv",testing="test.csv", train_percentage= train_percentage)

	def visual(self):
		colormap = plt.cm.viridis
		plt.figure(figsize=(11,9))
		plt.title('Pearson Correlation of Features', y=1.05, size=15)
		sns.heatmap(self.train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
		plt.show()

	def test_models(self):

		logreg = LogisticRegression()
		logreg.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = logreg.predict(self.test)
		acc_log = round(logreg.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_log

		svc = SVC()
		svc.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = svc.predict(self.test)
		acc_svc = round(svc.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_svc

		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = knn.predict(self.test)
		acc_knn = round(knn.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_knn

		gaussian = GaussianNB()
		gaussian.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = gaussian.predict(self.test)
		acc_gaussian = round(gaussian.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_gaussian

		perceptron = Perceptron()
		perceptron.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = perceptron.predict(self.test)
		acc_perceptron = round(perceptron.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_perceptron

		linear_svc = LinearSVC()
		linear_svc.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = linear_svc.predict(self.test)
		acc_linear_svc = round(linear_svc.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_linear_svc

		sgd = SGDClassifier()
		sgd.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = sgd.predict(self.test)
		acc_sgd = round(sgd.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_sgd

		decision_tree = DecisionTreeClassifier()
		decision_tree.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = decision_tree.predict(self.test)
		acc_decision_tree = round(decision_tree.score(self.X_test_cv, self.y_test_cv) * 100, 2)
		acc_decision_tree

		random_forest = RandomForestClassifier(n_estimators=100)
		random_forest.fit(self.X_train_cv, self.y_train_cv)
		Y_pred = random_forest.predict(self.test)
		random_forest.score(self.X_train, self.Y_train)
		acc_random_forest = round(random_forest.score(self.X_test_cv, self.y_test_cv) * 100, 2)
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

	## same as above, but use sklearn cross validation library
	def test2(self):
		from sklearn.model_selection import cross_val_score
		
		clf = SVC(kernel = 'linear', C = 1)
		scores_SVS = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_SVS = np.mean(scores_SVS)


		clf = LogisticRegression()
		scores_LR = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_LR = np.mean(scores_LR)

		clf = RandomForestClassifier(n_estimators=100)
		scores_RF = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_RF = np.mean(scores_LR)

		clf = KNeighborsClassifier(n_neighbors = 10)
		scores_KNN = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_KNN = np.mean(scores_KNN)

		clf = GaussianNB()
		scores_GNB = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_GNB = np.mean(scores_GNB)

		clf = Perceptron()
		scores_PCT = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_PCT = np.mean(scores_PCT)

		clf = SGDClassifier()
		scores_SGD = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_SGD = np.mean(scores_SGD)

		clf = DecisionTreeClassifier()
		scores_DT = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_DT = np.mean(scores_DT)

		from xgboost import XGBClassifier
		clf = XGBClassifier()
		scores_XGBC = cross_val_score(clf, self.X_train, self.Y_train, cv = 5)
		scores_XGBC = np.mean(scores_XGBC)

		models = pd.DataFrame({
	    	'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'XGBoost Classifier'],
	    	'Score': [scores_SVS, scores_KNN, scores_LR, scores_RF, scores_GNB, scores_PCT, scores_SGD, scores_DT, scores_XGBC]})
		return models.sort_values(by='Score', ascending=False)
		#.sort_values(by='Score', ascending=False)

class Combine_Models(Train_Model):
	def __init__(self, training="train.csv",testing="test.csv", train_percentage=.95):
		Data.__init__(self, training="train.csv",testing="test.csv", train_percentage= train_percentage)
		self.train_all()

	def train_all(self):
		self.clf_SVC = SVC(kernel = 'linear', C = 1).fit(self.X_train_cv, self.y_train_cv)
		self.clf_LR = LogisticRegression().fit(self.X_train_cv, self.y_train_cv)
		self.clf_RF = RandomForestClassifier(n_estimators=100, max_leave_nodes=12, max_depth=12).fit(self.X_train_cv, self.y_train_cv)
		self.clf_KNN = KNeighborsClassifier(n_neighbors = 10).fit(self.X_train_cv, self.y_train_cv)
		self.clf_GNB = GaussianNB().fit(self.X_train_cv, self.y_train_cv)
		self.clf_PCT = Perceptron().fit(self.X_train_cv, self.y_train_cv)
		self.clf_SGD = SGDClassifier().fit(self.X_train_cv, self.y_train_cv)
		self.clf_DT = DecisionTreeClassifier().fit(self.X_train_cv, self.y_train_cv)
		self.clf_XGB = XGBClassifier().fit(self.X_train_cv, self.y_train_cv)

	def predict_test(self):
		self.predictSVC = self.clf_SVC.predict(self.X_test_cv)
		self.predictLR = self.clf_LR.predict(self.X_test_cv)
		self.predictRF = self.clf_RF.predict(self.X_test_cv)
		self.predictKNN = self.clf_KNN.predict(self.X_test_cv)
		self.predictGNB = self.clf_GNB.predict(self.X_test_cv)
		self.predictPCT = self.clf_PCT.predict(self.X_test_cv)
		self.predictSGD = self.clf_SGD.predict(self.X_test_cv)
		self.predictDT = self.clf_DT.predict(self.X_test_cv)
		self.predictXGB = self.clf_XGB.predict(self.X_test_cv)
		self.combine = pd.DataFrame({ 'SVC': self.predictSVC,'SGD': self.predictSGD,'LR': self.predictLR,'RF': self.predictRF,'KNN': self.predictKNN,'GNB': self.predictGNB,'PCT': self.predictSGD, 'DT': self.predictDT, 'XBG': self.predictXGB})
		self.combine['mean'] = self.combine.mean(axis=1)
		self.combine['mean2'] = self.combine['mean'].round(0)
		self.combine['actual'] = self.y_test_cv.values

	def predict_final(self):
		predictSVC = self.clf_SVC.predict(self.test)
		predictLR = self.clf_LR.predict(self.test)
		predictRF = self.clf_RF.predict(self.test)
		predictKNN = self.clf_KNN.predict(self.test)
		predictGNB = self.clf_GNB.predict(self.test)
		predictPCT = self.clf_PCT.predict(self.test)
		predictSGD = self.clf_SGD.predict(self.test)
		predictDT = self.clf_DT.predict(self.test)
		predictXGB = self.clf_XGB.predict(self.test)
		combine = pd.DataFrame({ 'SVC': predictSVC,'SGD': predictSGD, 'LR': predictLR,'RF': predictRF,'KNN': predictKNN,'GNB': predictGNB,'PCT': predictSGD, 'DT': predictDT, 'XBG': predictXGB})
		combine['mean'] = combine.mean(axis=1)
		combine['mean2'] = combine['mean'].round(0)


		StackingSubmission = pd.DataFrame({ 'PassengerId': self.PassengerId,'Survived': combine['mean2'].astype(int)})
		StackingSubmission.to_csv("StackingSubmission.csv", index=False)


### The following is coppied pretty exactly from https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Some useful parameters which will come in handy later on

# # Class to extend the Sklearn classifier
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
        return (self.clf.fit(x,y).feature_importances_)


class First_Level_Model(Data, SklearnHelper):
 	def __init__(self, training="train.csv",testing="test.csv"):
		Data.__init__(self, training="train.csv",testing="test.csv")
		self.y_train = self.train['Survived'].ravel()
		self.train = self.train.drop(['Survived'], axis=1)
		self.x_train = self.train.values # Creates an array of the train data
		self.x_test = self.test.values # Creats an array of the test data
		# Some useful parameters which will come in handy later on
		self.ntrain = self.x_train.shape[0]
		self.ntest = self.x_test.shape[0]
		self.SEED = 0 # for reproducibility
		self.NFOLDS = 5 # set folds for out-of-fold prediction
		self.kf = KFold(self.ntrain, n_folds= self.NFOLDS, random_state=self.SEED)


		# Put in our parameters for said classifiers
		# Random Forest parameters
		self.rf_params = {
		    'n_jobs': -1,
		    'n_estimators': 500,
		     'warm_start': True, 
		     #'max_features': 0.2,
		    'max_depth': 6,
		    'min_samples_leaf': 2,
		    'max_features' : 'sqrt',
		    'verbose': 0
		}
		# Extra Trees Parameters
		self.et_params = {
		    'n_jobs': -1,
		    'n_estimators':500,
		    #'max_features': 0.5,
		    'max_depth': 8,
		    'min_samples_leaf': 2,
		    'verbose': 0
		}

		# AdaBoost parameters
		self.ada_params = {
		    'n_estimators': 500,
		    'learning_rate' : 0.75
		}

		# Gradient Boosting parameters
		self.gb_params = {
		    'n_estimators': 500,
		     #'max_features': 0.2,
		    'max_depth': 5,
		    'min_samples_leaf': 2,
		    'verbose': 0
		}

		# Support Vector Classifier parameters 
		self.svc_params = {
		    'kernel' : 'linear',
		    'C' : 0.025
		    }


	def get_oof(self, clf, x_train, y_train, x_test):
	    oof_train = np.zeros((self.ntrain,))
	    oof_test = np.zeros((self.ntest,))
	    oof_test_skf = np.empty((self.NFOLDS, self.ntest))

	    for i, (train_index, test_index) in enumerate(self.kf):
	        x_tr = x_train[train_index]
	        y_tr = y_train[train_index]
	        x_te = x_train[test_index]

	        clf.train(x_tr, y_tr)

	        oof_train[test_index] = clf.predict(x_te)
	        oof_test_skf[i, :] = clf.predict(x_test)

	    oof_test[:] = oof_test_skf.mean(axis=0)
	    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


	def create_classifiers(self):
		self.rf = SklearnHelper(clf=RandomForestClassifier, seed=self.SEED, params=self.rf_params)
		self.et = SklearnHelper(clf=ExtraTreesClassifier, seed=self.SEED, params=self.et_params)
		self.ada = SklearnHelper(clf=AdaBoostClassifier, seed=self.SEED, params=self.ada_params)
		self.gb = SklearnHelper(clf=GradientBoostingClassifier, seed=self.SEED, params=self.gb_params)
		self.svc = SklearnHelper(clf=SVC, seed=self.SEED, params=self.svc_params)

	def create_OOF(self):
		self.create_classifiers()
		print 'a'
		self.et_oof_train, self.et_oof_test = self.get_oof(self.et, self.x_train, self.y_train, self.x_test) # Extra Trees
		print 'b'
		self.rf_oof_train, self.rf_oof_test = self.get_oof(self.rf,self.x_train, self.y_train, self.x_test) # Random Forest
		print 'c'
		self.ada_oof_train, self.ada_oof_test = self.get_oof(self.ada, self.x_train, self.y_train, self.x_test) # AdaBoost 
		print 'd'
		self.gb_oof_train, self.gb_oof_test = self.get_oof(self.gb,self.x_train, self.y_train, self.x_test) # Gradient Boost
		print 'e'
		self.svc_oof_train, self.svc_oof_test = self.get_oof(self.svc,self.x_train, self.y_train, self.x_test) # Support Vector Classifier
		print 'done'

class Feature_Importance(First_Level_Model):
	def __init__(self, First_Level_Model):
		self.First_Level = First_Level_Model
		self.dataframe()

	def dataframe(self):
		rf_features = self.First_Level.rf.feature_importances(self.First_Level.x_train,self.First_Level.y_train)
		et_features = self.First_Level.et.feature_importances(self.First_Level.x_train, self.First_Level.y_train)
		ada_features = self.First_Level.ada.feature_importances(self.First_Level.x_train, self.First_Level.y_train)
		gb_features = self.First_Level.gb.feature_importances(self.First_Level.x_train,self.First_Level.y_train)
		cols = self.First_Level.train.columns.values

		self.feature_dataframe = pd.DataFrame( {'features': cols, 'Random Forest feature importances': rf_features,'Extra Trees  feature importances': et_features,'AdaBoost feature importances': ada_features,'Gradient Boost feature importances': gb_features})
		self.feature_dataframe['mean'] = self.feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

		self.fbase_predictions_train = pd.DataFrame( {'RandomForest': self.First_Level.rf_oof_train.ravel(),
 		    'ExtraTrees': self.First_Level.et_oof_train.ravel(),
  		   'AdaBoost': self.First_Level.ada_oof_train.ravel(),
   		   'GradientBoost': self.First_Level.gb_oof_train.ravel()
   		 })

	def train_second(self):
		x_train = np.concatenate((self.First_Level.et_oof_train, self.First_Level.rf_oof_train, self.First_Level.ada_oof_train, self.First_Level.gb_oof_train, self.First_Level.svc_oof_train), axis=1)
		x_test = np.concatenate((self.First_Level.et_oof_test, self.First_Level.rf_oof_test, self.First_Level.ada_oof_test, self.First_Level.gb_oof_test, self.First_Level.svc_oof_test), axis=1)

		gbm = xgb.XGBClassifier(
		    #learning_rate = 0.02,
		 n_estimators= 2000,
		 max_depth= 4,
		 min_child_weight= 2,
		 #gamma=1,
		 gamma=0.9,                        
		 subsample=0.8,
		 colsample_bytree=0.8,
		 objective= 'binary:logistic',
		 nthread= -1,
		 scale_pos_weight=1).fit(self.First_Level.x_train, self.First_Level.y_train)
		predictions = gbm.predict(self.First_Level.x_test)


		StackingSubmission = pd.DataFrame({ 'PassengerId': self.First_Level.PassengerId,'Survived': predictions })
		StackingSubmission.to_csv("StackingSubmission.csv", index=False)


