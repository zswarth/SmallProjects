import matplotlib.pyplot as plt 	
import numpy as np

x_set = [0,-2,-1,4,3,2,1,2,-3,1,-2,3,8,7,6,5,6,7,6,8,9,11,14,13,11,8,6]
y_set = [12,12,11,14,13,12,11,12,13,11,12,13,1,2,5,2,3,2,1,1,2,1,4,3,1,8,6]
labels = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def graph(x,y,labels):
	n = len(labels)
	for i in range(0,n):
		if labels[i] == 1:
			plt.scatter(x[i],y[i], marker = '^')
	for i in range(0,n):
		if labels[i] == 0:
			plt.scatter(x[i],y[i], marker = 's', color = 'r')
	plt.show()

def unit_step(number):
	if number>0:
		return 1
	else:
		return 0

def perceptron_train(x1,x2,w0,w1,w2,n,iteration):
	for i in range(0,iteration):
		 add = w0+(w1*x1[i])+(w2*x2[i])
		 error=labels[i]-unit_step(add)
		 if unit_step(add)!=labels[i]:
		 	w1=w1+n*error*x1[i]
		 	w2=w2+n*error*x2[i]
		 	w0=w0+n*error*1
		 elif unit_step(add)==labels[i]:
		 	pass
		 
	return w0,w1,w2


def show_line(w,datax, datay, values):
	for i in range(0,len(datax)):
		if values[i] == 0:
			plt.scatter(datax[i], datay[i], marker = ">")
		else:
			plt.scatter(datax[i], datay[i], marker = '+')
	xs = np.arange(0,10, .1)
	ys = (-w[0]-(w[1]*xs))/w[2]
	plt.plot(xs,ys)
	plt.show()
a = perceptron_train(x_set,y_set,0,3,2,0.2,len(labels))
print a
show_line(a, x_set, y_set, labels)
graph(x_set,y_set,labels)