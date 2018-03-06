
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn.model_selection import train_test_split

# data = np.loadtxt('train_MNIST.csv', dtype = str, delimiter = ',')
# y = np.asarray(data[1:, 0:1], dtype='float')
# X = np.asarray(data[1:,1:], dtype='float')

class Perceptron(object):
	def __init__(self, data, labels, alpha = .1, iteration = 30, test_percentage = .9):
		self.X = self.add_ones(data)
		self.y = labels
		self.alpha = alpha
		self.iteration = iteration
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_percentage, random_state=42)
		self.weights = self.all_numbers(self.X_train, self.y_train)
		self.accuracy = self.test_all(self.X_test, self.y_test, self.weights)


	def add_ones(self, x):
	 	a, b = np.shape(x)
		c = np.ones((a , 1))   
		return np.hstack((c, x))

	def img(self, row, data):

		image = np.zeros((28,28))
		for i in range(0,28):
			for j in range(0,28):
				pix = 28*i+j
				image[i,j] = data[row, pix]
		plt.imshow(image, cmap = 'gray')
		plt.show()

	def create_weights(self, data):
		a, b = np.shape(data)
		weights = np.random.rand(b,1)
		return weights

	def predict(self, data_point, weights):
		b = np.dot(data_point, weights)
		a = b>0
		return a*1

	def one_number(self, labels, number):
		return (labels == number)*1

	def update(self, weights, data_point, labels, alpha=.1):
		predicted = self.predict(data_point, weights)
		weight_temp = np.zeros(np.shape(weights))
		weight_temp[:,0] = alpha*(labels-predicted)*data_point
		return weight_temp+weights

	def train_perceptron(self, data, labels, weights, alpha = .1, iterations = 100):
		for j in range(0, iterations):
			for i in range(0, len(data)):
				weights = self.update(weights, data[i], labels[i], alpha)
		return weights

	def test_perceptron_f(data, labels, weights):
	    a,b = np.shape(data)
	    predicted = self.predict(data, weights)
	    correct = (predicted==labels)*1==1
	    true_pos = np.sum((labels==1)*(correct))
	    true_neg = np.sum((labels==0)*(correct))
	    tp_p = true_pos/float(np.sum(labels))
	    print np.sum(labels)
	    tn_p = true_neg/float(a- np.sum(labels))
	    return true_pos, true_neg, tp_p, tn_p, a
	    
	def all_numbers(self, data,labels):
		c,d = np.shape(data)
		w = self.create_weights(data)
		weights = []
		for i in range(0,  len(np.unique(labels))):
			z = self.one_number(labels, i)
			a = self.train_perceptron(data, z, w, .1, 4)
			weights.append(a[:,0])
		return np.asarray(weights)

	def one_all(self, data, weights):
		a = np.dot(data,np.transpose(weights))
		b = len(np.shape(data))
		if b == 1:
			return np.argmax(a)
		return np.argmax(a, axis=1)


	def test_all(self, data, labels, weights):
		a, b = np.shape(labels)
		predicted = self.one_all(data, weights)
		correct = predicted == labels[:,0]
		accuracy = np.sum(correct)/float(a)
		return accuracy


