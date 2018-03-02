import numpy as np
import matplotlib.pyplot as plt

#Iris Data Set

flowers = np.loadtxt('Iris.csv', dtype=str, delimiter=',')
train = np.asarray(flowers[1:,1:5], dtype=float)
labels = flowers[1:, 5]

def make_labels_numeric(labels):
	flw = np.asarray(np.unique(labels), dtype=str)
	for i in range(0, len(labels)):
		labels[i] = np.where(flw==labels[i])[0][0]
	return np.asarray(labels, dtype=int)

labels = make_labels_numeric(labels)

#Instead of writing loops, we're going to want to just multiply vectors.  It's way faster
#We're going to need a 1's place for the constant.

#incert graphic here

def add_odes(x):
	m,n = np.shape(x)
	temp = np.ones((m,n+1))
	temp[:,1:n+1] = x
	return temp

train = add_odes(train)
a = np.unique(labels)

def graph(data, labels, d1=1, d2=2):
	colors = ['red','black','green', 'blue','orgage']
	mkr = ['.', '<', '>', ',']
	for i in range(0, len(labels)):
		plt.scatter(data[i,d1], data[i,d2], marker = mkr[labels[i]], color = colors[labels[i]])
	plt.show()

#show a few graphs
# print graph(train, labels, d1=2, d2=3)


#initalize with random weights
def create_weights(data):
	m,n = np.shape(data)
	weights = np.random.rand((n))
	return np.array(weights)
	#write a funtion that will re


#explain the full weights here

w = create_weights(train)

def predict(data_point, weights):
	a = np.dot(weights, data_point)
	if a>0:
		return 1
	else:
		return 0
	#for any given data point, return the predicted value (shoudl be 0 or 1 for a binary classification)



def train_perceptron(data, labels, weights, alpha = .1, iterations = 100):
	m,n = np.shape(data)
	weight_history = np.zeros((iterations, np.shape(weights)[0]))
	weight_history[0] = weights
	weight_history[0] = weights
	for b in xrange(iterations):
		for i in range(0,m):
			error = labels[i] - predict(weights,data[i])
			for j in range(0, n):
				weights[j] = weights[j]+alpha*error*data[i,j]
		weight_history[b] = weights
	return weights, weight_history

w,b = train_perceptron(train[0:100],labels[0:100], w)
print train[1]


def test_percepton(data, labels, weights):
	error = 0
	for i in range(0, len(labels)):
		p = predict(data[i], weights)
		error += abs(p-labels[i])
	return 100 - error/float(len(labels))*100



def graph_line(weights, data, labels, d1=1, d2=2):
	colors = ['red','black','green', 'blue','orgage']
	mkr = ['.', '<', '>', ',']
	for i in range(0, len(labels)):
		plt.scatter(data[i,d1], data[i,d2], marker = mkr[labels[i]], color = colors[labels[i]])
	x = np.arange(np.min(data[:, d1]), np.max(data[:,d1]),.01)
	y = (-weights[0]-weights[d1]*x)/weights[d2]
	plt.plot(x,y)
	plt.show()

print test_percepton(train[0:100], labels[0:100],w)
graph_line(w, train[0:100], labels[0:100], d1=1, d2=4)

