import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import Perceptron


class BinaryPredictor(object):
	
	def __init__(self, training, test):
		self.clf = None
		self.cv = None
		self.dv = None
		self.X, self.Y = self.pre_process(training)
		self.test_X, self.test_Y = self.pre_process(test)



	def pre_process(self, text):
		if not self.cv:
			self.cv = CountVectorizer()
			self.cv.fit(text.keys())
		feature_dict = self.extract_features(text.keys())
		if not self.dv:
			self.dv = DictVectorizer()
			self.dv.fit(feature_dict)
		X_additional = self.dv.transform(feature_dict)
		X_bow = self.cv.transform(text.keys())
		X = np.hstack([X_bow.toarray(), X_additional.toarray()])
		Y = text.values()
		return X,Y


	def extract_features(self, text):
		tweets = []
		for i in text:
			tweets.append(dict(length=len(i.split())))
		return tweets

	def train(self):
		self.clf.fit(self.X, self.Y)

	def predict(self, text):
		feature, _ = self.pre_process({t:0 for t in text})
		return map(bool, self.clf.predict(feature))

	def evaluate(self):
		return cross_val_score(self.clf, self.X, np.array(self.Y))




class PerceptronPredictor(BinaryPredictor):

	def __init__(self, training, test):
		super(PerceptronPredictor, self).__init__(training, test)
		self.clf = Perceptron(shuffle = True)