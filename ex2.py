import scipy as sci
import numpy as num
import matplotlib.pyplot as plt
from scipy import e
from scipy import log


class Regression(object):
	def __init__(self):
		self.data = num.loadtxt('Example 2 ML Class Python/ex2data1.txt', delimiter= ',')
		self.X = self.data[:, 0:2]
		self.y = self.data[:, 2:3]

	def Plot(Self, X, y):
		pos = num.zeros((0,2))
		neg = num.zeros((0,2))
		for i in range(0,len(X)):
			if y[i] == 1:
				neg = num.vstack((neg, X[i,0:2]))
		for i in range(0,len(X)):
			if y[i] == 0:
				pos = num.vstack((pos, X[i,0:2]))
		plt.scatter(neg[:,0], neg[:,1])
		plt.scatter(pos[:,0], pos[:,1],c = 'y')
		plt.show()


	def Sigmoid(self, z):
		g = num.zeros(num.size(z))
		if type(z) == int:
			return 1/(1+e**(-z))
		else:
			a = num.size(z)
			for i in range(0, a):
					g[i] = 1/(1+e**(-z[i]))
			return g


	def costFunction(self, theta, X, y):
		m = len(y)
		J = 0
		grad = num.zeros(num.shape(theta))

		z = num.dot(X, theta)
		h = self.Sigmoid(z)
		J = (-1./m)*(num.dot(num.transpose(log(h)),y)+num.dot(num.transpose(log(1-h)),(1-y)))

		grad = (1./m)*(num.dot(num.transpose(X),(h-y)))

		return J, grad


	def Cost(self, init_theta):
		[m,n] = num.shape(self.X)
		a = num.ones((m,1))
		X = num.hstack((a, self.X))
		[cost, grad] = self.costFunction(init_theta, self.X, self.y)
		#print "Cost at initial theta (zeros)", cost
		#print "Gradient at inital theta", grad
		return cost
