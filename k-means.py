from numpy import genfromtxt
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

#note -- I use the term 'cluster' in this document.  I realized in retrospect, 'centroid'
# is a better term.


## This is the data we are going ot use for our project
## The first column is are x-values, and the second column is our y-values
# data = genfromtxt('k-means data.csv', delimiter=',')
data = genfromtxt('k-means data2.csv', delimiter=',')


# 1) Graph this data.  There are no lables, so we can just graph it all together


plt.scatter(data[:,0], data[:,1])
# plt.show()


# 2)
# We are going to need a distance funtion.  You can pull
# this streight from your last assignment

def distance(point1, point2):
	dis = point1-point2
	return (np.dot(dis, dis))**.5



# 3) Create Arbitrary first centroid.
# We'll write this just for 2 dimentions.
# This funtion should create a 2 d array with each row being a 2-d centroid
# of random coordiantes.  The coorediates should automatically be between the min 
# and the max of the data.  IE: if the x-range of a set of data is between -20 and 30, you should
# choose random x- values between -20 and 30
# Hint: If you want to turn a normal list into a numpy array: the code is
# np.asarray(list)

def create_centroids(data, k):
	centroids = []
	maxx = int(max(data[:,0]))
	minx = int(min(data[:,0]))
	maxy = int(max(data[:,1]))
	miny = int(min(data[:,1]))
	for i in range(0,k):
		centroids.append([randrange(minx, maxx), randrange(miny, maxy)])
	return np.asarray(centroids)


### 4) Each line of our data is going to need to be assigned a 
### label representing the nearest centroid.
### Create a funtion that will output a list of labels that coresponds to each line in our data
#### REMEBER: TO USE THS, YOU WILL NEED TO CREATE A SET OF CENTROIDS
#### CREATE THIS BEFORE YOU TEST THIS FUNTION

def label(data, centroids):
	m, n = np.shape(data)
	lab = []
	for i in range(0, m):
		dis = []
		x, y = np.shape(centroids)
		for j in range(0,x):
			dis.append(distance(centroids[j,:], data[i, :]))
		lab.append(np.argmax(dis))
	return np.asarray(lab)

centroids = create_centroids(data, 3)
labels = label(data, centroids)


### This is data we are goning to want to visuallize
### Create a funtion that will graph the centroids (use a notacably different symbol)
### as well as our data.  Now we need to color our data differenly for each centroids
#### HINTS:
	# To use an arbitrayr number of colors, you an use the following array, and just choose
	# an index in that array.
	# colors = ['b','g','r','c','m','y','k','w']
	# Example -  to scatter red: plt.scatter(x, y, color = colors[2])


def graph(data, labels, centroids):
	colors = ['b','g','r','c','m','y','k','w']
	for i in range(0, max(labels)+1):
		plt.scatter(centroids[i,0],centroids[i,1], color = colors[i+1], marker = '<', s = 100)
		for j in range(0, len(labels)):
			if labels[j] == i:
				plt.scatter(data[j,0], data[j,1], color = colors[i])
	plt.show()



#### Once we lable our data, we need to update the coordinates of our centroids
#### Write a funtion that takes our labels, and returns the coordiates of our updated centroids


def update_centroids(data, labels, centroids):
	m, n = np.shape(centroids)
	for i in range(0, m):
		x_total = 0.0
		y_total = 0.0
		count = 0
		for j in range(0,len(labels)):
			if labels[j] == i:
				x_total += data[j, 0]
				y_total += data[j, 1]
				count += 1.0
		if count != 0:
			centroids[i] = (x_total/count, y_total/count)
	return centroids

for i in range(0,8):
	labels = label(data, centroids)
	centroids= update_centroids(data, labels, centroids)
	graph (data, labels, centroids)
