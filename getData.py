#### Class 1
		## Get and Clean Data

## Class 2
		## Classifier

### Class 3
		### Display Set of Training Data
		#### Display arbitrary Test Images in Classification

###Function that prints digits

from sklearn import datasets
from sklearn import svm
import pylab as pl
import numpy

class getData(object):
	pass


class DigitPredictor(object):

	def __init__(self, training, labels, test):
		self.clf = None
		self.data = None
		self.target = None
		self.test = None

	def train(self):
		self.clf.fit(self.data, self.target)

	def predict(self, data):
		return self.clf.predict(data)

	def error_rate(self):
		error = 0
		y = self.test[:,0]
		x = self.test[:,1:750]
		for i in range(1, len(y)):
			if self.clf.predict(x[i,:]) != y[i]:
				error += 1
		return ("error rate is ", (error/len(y))*100, "%")


	def display(self, data, number):
		digits = data
		pl.figure(1, figsize=(3,3))
		pl.imshow(digits.images[number], cmap=pl.cm.gray_r, interpolation='nearest')
		pl.show()



class SVC(DigitPredictor):

	def __init__(self, training, labels, test):
		super(SVC, self).__init__(training, labels, test)
		self.clf = svm.SVC(gamma=.001, C=100)
		self.data = training
		self.target = labels
		self.test = test

class Data(object):

	def __init__(self):
		self.data = datasets.load_digits()
		self.training = self.data.data
		self.labels = self.data.target
		self.test = self.training[-1]

class Data2(object):

	def __init__(self):
		self.data = numpy.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)
		self.training = self.data[0:20000,1:750]
		self.labels = self.data[0:20000,0]
		self.test = self.data[20001:42000,:]