from numpy import genfromtxt
import matplotlib.pyplot as plt
from random import randrange
import numpy as np



iris = np.genfromtxt('iris.csv', dtype=str, delimiter = ',')
iris_data = np.asarray(iris[1:,0:4], dtype=float)
iris_labels = iris[1:,4]

plt.scatter(iris_data[0:50,0], iris_data[0:50,1], color = 'red')
plt.scatter(iris_data[50:100,0], iris_data[50:100,1], color = 'red')
plt.scatter(iris_data[100:150,0], iris_data[100:150,1], color = 'red')

# plt.show()

#The Iris data set with 3 labels is very difficult to classifiy with clustering.  We'll go back to this
#when i work on supervisted learning.  The iris-Versicolor and the Iris-Virginica have signifigant overlap that is hard to cluster

#We're only looking at the first two attributes in these graphs
#Sepal Lenght and Sepal Width

# plt.scatter(iris_data[0:50,2], iris_data[0:50,3], color = 'red')
# plt.scatter(iris_data[50:100,2], iris_data[50:100,3], color = 'blue')
# plt.scatter(iris_data[100:150,2], iris_data[100:150,3], color = 'green')

# plt.show()

#Looking at other dementions, we can see that there is still overlap.

#To start, lets just look at Setosa and Versicolor which are well seperated.


#K Means is unsupervised though - we're going to group our data without any labels.

# plt.scatter(iris_data[0:100,0], iris_data[0:100,1], color = 'red')
# plt.show()

#Firstly, we need to write a funtion to find the euclidian distance between any two points.

def distance(point1, point2):
	dis = point1-point2
	return (np.dot(dis, dis))**.5


# # Create Arbitrary first centroid.
## More efficient k-means clustering use more complex methods to initiate the centroid.

def create_centroids(data, num_clusters = 3):
	centroids = []
	a,b = np.shape(data)
	for j in range(0, num_clusters):
		temp = []
		for i in range(0, b):
			maxx = int(max(data[:,i]))
			minx = int(min(data[:,i]))
			temp.append(randrange(minx, maxx))
		centroids.append(temp)
	return np.asarray(centroids)

centroids = create_centroids(iris_data[0:100], num_clusters=2)

# Each line of our data is going to need to be assigned a label representing the nearest centroid.

def label(data, centroids):
	m, n = np.shape(data)
	lab = []
	for i in range(0, m):
		dis = []
		x, y = np.shape(centroids)
		for j in range(0,x):
			dis.append(distance(centroids[j,:], data[i, :]))
		lab.append(np.argmin(dis))
	return np.asarray(lab)



labels = label(iris_data[0:100], centroids)


def graph(data, labels, centroids):
	colors = ['b','g','r','c','m','y','k','w']
	for i in range(0, max(labels)+1):
		plt.scatter(centroids[i,0],centroids[i,1], color = colors[i+2], marker = '<', s = 150)
		for j in range(0, len(labels)):
			if labels[j] == i:
				plt.scatter(data[j,0], data[j,1], color = colors[i])


# # graph(iris_data[0:100], labels, centroids)
# print np.shape(centroids)
# print centroids
# #### Once we lable our data, we need to update the coordinates of our centroids
# #### Write a funtion that takes our labels, and returns the coordiates of our updated centroids



def update_centroids(data, labels, centroids):
	m,n = np.shape(centroids)
	temp_centroid = np.zeros((m,n))
	count = np.zeros((m,n))
	for j in range(0, len(labels)):
	 	for k in range(0, n):
			temp_centroid[labels[j],k]+= data[j,k]
			count[labels[j],k]+= 1
	for i in range(0,m):
		for j in range(0,n):
			if count[i,j] == 0:
				count[i,j] += 1
	return temp_centroid/count


def graph_axis(data, labels, centroids, axis):
	colors = ['b','g','r','c','m','y','k','w']
	for i in range(0, max(labels)+1):
		axis.scatter(centroids[i,0],centroids[i,1], color = colors[i+2], marker = '<', s = 150)
		for j in range(0, len(labels)):
			if labels[j] == i:
				axis.scatter(data[j,0], data[j,1], color = colors[i])

			

def final_algorithm(data, n_clusters = 2, iter = 3):
	centroids = create_centroids(data, num_clusters = n_clusters)	
	labels = label(data, centroids)
	f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
	graph_axis(data, labels, centroids, ax1)
	plots = [ax2, ax3, ax4]
	for i in plots:
		centroids = update_centroids(data, labels, centroids)
		labels = label(data, centroids)
		graph_axis(data, labels, centroids, i)
	plt.show()
	for i in range(0,iter-3):
		centroids = update_centroids(data, labels, centroids)
		labels = label(data, centroids)
	return centroids, labels

final_algorithm(iris_data[0:100])
