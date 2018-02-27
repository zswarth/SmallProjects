#### Introduction to the Perceptron. 
#### Fill In ALL QUESTIONS


#__________DAY 1______________

##Questoins

### 1) What is the difference between Supervised and Unspervised Learning


### 2) What is the difference bewteen Regression and Classification Problems




#### Project

## We have two data sets.  All our work needs to be done for AN ARBITRARY NUMBER OF DIMENTIONS.  The firt Data set is in two dimentions.  The second is in 4 (This is the famous Iris Data Set)

import numpy as np

#Data Set 1: 2D Data
Labels = np.array([0, 1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,0,1])
data = np.array([[29,126,58,29,94,255,51,71,90,280,282,283,86,229,57,48,194,174,18,196,80,22,55,133,249,218,144,86,258,238,28,0,109,38,195,38,121],[120,51,158,176,83,88,143,161,116,57,116,92,173,59,167,183,24,74,107,109,181,195,167,66,47,27,02,07,20,29,155,137,102,135,58,147,1]])
data = data.transpose()

Labels_test = np.array([0,1,0,0,1,1,0,0,0,1,1])


data_test = np.array([[90,194,1,24,195,111,53,90,99,193,208], [141,89,126,178,118,96,168,179,163,65,62]])
data_test = data_test.transpose()


#Data Set 2: Iris FLower Data

#Make sure you have Iris.txt downloaded

#flowers = np.loadtxt('Iris.txt', dtype=str)
#flowers = np.asfarray(flowers[1:, :], dtype=float)

# We're going to work with this data set later.


def add_odes(x):
	#Write a funtion that addes a collumn of ones to the array
	m,n = np.shape(x)
	temp = np.ones((m,n+1))
	temp[:,1:n+1] = x
	return temp


data = add_odes(data)
data_test = add_odes(data_test)



def graph(data, labels):
	import matplotlib.pyplot as plt
	for i in range(0, len(labels)):
		if labels[i] == 0:
			plt.scatter(data[i,1], data[i,2], marker = '.', color = 'red')
		else:
			plt.scatter(data[i,1], data[i,2], marker = '<', color = 'black')
	plt.show()

# graph(data_test, Labels_test)

#write a funtion that will plot the first two dimentions of the data. Make sure different shapes/colors are used for each label.



#______________Day 2___________________

def create_weights(data):
	m,n = np.shape(data)
	weights = np.random.rand((n))
	return np.array(weights)
	#write a funtion that will return a numpy array of RANDOM WEIGHTS.  The array should be the correct length for the data given.

a = create_weights(data)


def predict(data_point, weights):
	a = np.dot(weights, data_point)
	if a>0:
		return 1
	else:
		return 0
	#for any given data point, return the predicted value (shoudl be 0 or 1 for a binary classification)

#_____________Day 3____________________


def train_perceptron(data, labels, weights, alpha = .01, iterations = 100):
	m,n = np.shape(data)
	for b in xrange(iterations):
		for i in range(0,m):
			error = labels[i] - predict(weights,data[i])
			for j in range(0, n):
				weights[j] = weights[j]+alpha*error*data[i,j]
	return weights

b = train_perceptron(data,Labels, a)
print b

	#Train a perceptron.  Return weights



#______________Day 4_________________

def test_percepton(data, labels, weights):
	error = 0
	for i in range(0, len(labels)):
		p = predict(data[i], weights)
		error += abs(p-labels[i])
		return 100 - error/float(len(labels))*100

print test_percepton(data_test, Labels_test, b)
	#Return percentage correct




