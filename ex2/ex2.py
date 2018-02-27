import numpy as np
import matplotlib.pyplot as plt
from math import exp



class ex2(object):
	def __init__(self):
		self.filename = 'ex2data1.txt'
		self.data = np.loadtxt(self.filename, dtype=float, delimiter=',')
		self.x = self.data[:, 0:2]
		self.y = self.data[:, 2]
		self.m, self.n = self.x.shape
		self.theta = np.zeros((3))


	def update_x(self):
		self.x = np.concatenate((np.ones((self.m, 1)),self.x), axis=1)
		self.m, self.n = self.x.shape
		initial_theta = np.zeros(self.n+1)

	def plot(self):
		for i in range(self.m):
			if self.y[i] == 0:
				plt.scatter(self.x[i,1], self.x[i,2], color = 'red')
			if self.y[i] == 1:
				plt.scatter(self.x[i,1], self.x[i,2], color = 'blue')		
		plt.show()

	def sigmoid(self, z):
		temp = 1+exp(-z)
		temp2 = 1/(temp)
		return temp2


	def cost(self):
		total = 0
		h= self.sigmoid(self.x.dot(self.theta))
		cost = sum(-y*np.log(h)-(1.0-y)*np.log(1.0-h))
		grad = self.x.T.dot(h-y)
		return cost, grad

