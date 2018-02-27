import numpy as np
import matplotlib.pyplot as plt

#x referes to the population size in 10,000s
# y referes to the profi in $10,000s

class ex1(object):

	def __init__(self):

		self.filename = 'ex1data1.txt'
		self.data = np.loadtxt(self.filename, dtype=float, delimiter=',')
		self.X = self.data[:,0]
		self.Y = self.data[:,1]
		self.m = np.size(self.Y) #number of training examples


###Function to Plot data
	def plot_data(self):
		plt.scatter(self.X, self.Y)
		plt.show()


###Gradient Decent
	## Update Equations
	def UpdateX(self):
		size = self.m
		array = np.ones((size,2))
		array[:,1] = self.X
		self.X = array

	def createTheta(self, interationss = 1500, alphaa = .01):
		self.theta = np.zeros((2,1)) ##initalize fitting paramters
		self.iterations = interationss
		self.alpha = alphaa
		
	def ComputeCost(self):
		J = 0
		for i in range(0,self.m):
			J += (self.theta[0]+(self.theta[1]*self.X[i,1])-self.Y[i])**2
		J = J/(2*self.m)
		return J



	def gradientDescent(self, interations = 1500):
		for i in range(1,interations):
			J = 0
			for j in range(0,self.m):
				J += (self.theta[0]+(self.theta[1]*self.X[j,1])-self.Y[j])
			K = 0
			for j in range(0,self.m):
				K += ((self.theta[0]+(self.theta[1]*self.X[j,1])-self.Y[j])*(self.X[j,1]))
			self.theta[0] = (self.theta[0] - (self.alpha*(1/float(self.m))*J))
			self.theta[1] = (self.theta[1] - (self.alpha*(1/float(self.m))*K))


	def Plot(self):
		self.line = np.zeros((100,2))
		for i in range(0, 100):
			self.line[i,0] = (5+(i*((22-5)/100.0)))
			self.line[i,1] = self.line[i,0]*self.theta[1]+self.theta[0]

		plt.scatter(self.line[:,0], self.line[:,1], color='blue')
		plt.scatter(self.X[:,1], self.Y, color='red')
		plt.show()

	def Predict(self, population1 = 3.5, population2 = 7):
		predict1 = [1, population1]*self.theta
		predict2 = [1, population2]*self.theta
		print 'For a population of', population1, 'thousand, we predict a profit of', predict1*10, 'thousand'
		print 'For a population of', population2, 'thousand, we predict a profit of', predict2*10, 'thousand'


	def Run(self):
		self.UpdateX()
		self.createTheta()
		self.ComputeCost()
		self.gradientDescent()
		print "Theta found by gradeint descent:", self.theta
		self.Plot()
		self.Predict()

#multi variable
class multi(object):

	def __init__(self):

		self.filename = 'ex1data2.txt'
		self.data = np.loadtxt(self.filename, dtype=float, delimiter=',')
		self.X = self.data[:,0:2]
		self.Y = self.data[:,1]
		self.m = np.size(self.Y) #number of training examples

	def scale(self):
		self.mean1 = np.mean(self.X[:,0])
		self.mean2 = np.mean(self.X[:,1])
		for i in range(1,self.m):
			self.X[i,0] = self.X[i,0] - self.mean1
		for i in range(1,self.m):
			self.X[i,1] = self.X[i,1] - self.mean2
		self.std1 = np.std(self.X[:,0])
		self.std2 = np.std(self.X[:,1])
		for i in range(1,self.m):
			self.X[i,0] = self.X[i,0]/self.std1
		for i in range(1,self.m):
			self.X[i,1] = self.X[i,1]/self.std2


	def UpdateX(self):
		array = np.ones((self.m,3))
		array[:,1:3] = self.X
		self.X = array

	def get_theta(self, interationss = 1500, alphaa = .01):
		self.theta = np.zeros((3,1)) ##initalize fitting paramters
		self.iterations = interationss
		self.alpha = alphaa

	def gradDecent(self):
		self.cost_history = np.zeros(self.iterations)



####Finish up multi variable at some point.



	