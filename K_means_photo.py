import numpy as num
import scipy as sci
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans

class kMeans(object):

	def _init_(self, n_clusters, photo):
		self.pictures = None
		self.labels = None
		self.centroids = None
		self.vec = None
		self.new_colors = None


	def photo(self, photos):
		self.pictures=misc.imread(photos)

	def show_photo(self):
		plt.imshow(self.pictures)
		plt.show()


	def color_vector(self):
		[a, b, c] = num.shape(self.pictures)
		vec = num.ones((a*b, 3))
		for i in range(0,a):
			for j in range(0,b):
				vec[i*b+j, :] = self.pictures[i, j]
		self.vec = vec


	def find_centers(self, NClus=10):
		kmeans = cluster.KMeans(n_clusters = NClus)
		kmeans.fit(self.vec)
		self.labels = kmeans.labels_
		self.centroids = kmeans.cluster_centers_

	def change_colors(self):
		number_clusters = len(self.centroids)
		[a,b,c] = num.shape(self.pictures)
		self.new_colors = self.vec
		size = len(self.labels)
		for i in range(0,size):
			for j in range(0,number_clusters):
				if self.labels[i] == j:
					self.new_colors[i] = self.centroids[j]

	def alter_picture(self):
		[a, b, c] = num.shape(self.pictures)
		for i in range(0,a):
			for j in range(0,b):
				self.pictures[i, j] = self.new_colors[i*b+j]

	def new_photo(self, pic, NClus=5):
		self.photo(pic)
		self.color_vector()
		self.find_centers(NClus)
		self.change_colors()
		self.change_colors()
		self.alter_picture()
		self.show_photo()
		
