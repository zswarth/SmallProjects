import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Ulam(object):
	def __init__(self, size = 99):
		self.spiral = np.zeros((0,0))
		self.size = 2*size-1

	def prime(self, x):
		primes = [True]*x
		primes[0] = False
		primes[1] = False
		for i in range(2,int(x**.5+int(x**.5)+2)):
			if primes[i]:
				for j in range(i, x//2+1):
					if i*j<x:
						primes[i*j] = False
		return primes




	def make_curve(self):

		x  = self.size/2
		y  = self.size/2
		num = 1
		pass_num = 1
		spiral = np.zeros((self.size, self.size))

		spiral[x, y] = num
		num += 1

		finished = False

		while not finished:

			#down
			for i in range(2*pass_num - 1):
				y+= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			
			#right
			for i in range(2*pass_num - 1):
				x+= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break

			#up
			for i in range(2*pass_num):
				y -= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			#left
			for i in range(2*pass_num):
				x -= 1
				spiral[x,y] = num
				num+= 1
				if num>self.size*self.size-5:
					finished = True
					break
			pass_num += 1


		return spiral

	def replace_curve(self, array):
		self.size = np.shape(array)
		primes = self.prime(self.size[0]*self.size[1])
		for i in range(0,self.size[0]):
			for j in range(0, self.size[1]):
				if primes[int(array[i,j])]:
					array[i,j] = 1
				else:
					array[i,j] = 0
		return array


	def show(self):
		width = self.size
		height = self.size
		self.spiral = self.make_curve()
		self.spiral = self.replace_curve(self.spiral)	
		plt.imshow(self.spiral, cmap = cm.Greys_r)
		plt.show()

a = Ulam(size=50)
a.show()
