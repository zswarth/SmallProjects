from random import choice
from numpy import dot, random

def unit_step(x):
	if x < 0:
		return 0
	if x >= 0:
		return 1

training_data = [([0,0,0], 0), ([0,1,1], 1),([1,0,1], 1), ([1,1,1], 1)]

w = random.rand(3)
errors = []
eta = .2
n = 100

for i in xrange(n):
	x, expected = choice(training_data)
	result = dot(w, x)
	error = expected - unit_step(result)
	errors.append(error)
	for i in range(0,3):
		w += x[i]*eta*error

for x, _ in training_data:
	results = dot(x, w)
	print results
	print unit_step(results)