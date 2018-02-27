import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, pi
from scipy.io.wavfile import write
from scipy.io.wavfile import read

a, b = read('wrong_week3.wav')
print a

print b[0:200]
c = xrange(0,1000)
plt.plot(c,b[0:1000])
plt.show()

def data(x, frequency = 440, amplitude = 20):
	return amplitude*sin(x*frequency*(2*pi))


def show_wave(function, time=.01):
	a = np.arange(0,time,(time/1000.0))
	b = []
	for i in a:
		b.append(function(i))
	plt.plot(a,b)
	plt.show()


def create_data(function, bit_depth=0.0, sample_rate=10000.0, time=2.0):
	if bit_depth == 0:
		time_array = np.arange(0,time,1/sample_rate)
		dt = function(time_array)
	return dt



sample_data = create_data(data)
# c = xrange(0,1000)
# plt.plot(c,sample_data[0:1000])
# plt.show()
	

def digital_encode(function, bit_depth = 16.0, sampling_rate = 24000.0, max_amp = 10.0, time = .0001):
	x = np.arange(0,time, 1/sampling_rate)
	bit_depth= 2**bit_depth
	step_size = 2*max_amp/bit_depth
	b = []
	for i in x:
		num = function(i)
		b.append((num//step_size)*step_size)
	plt.plot(x,b)


	a = np.arange(0,time,(time/1000.0))
	b = []
	for i in a:
		b.append(function(i))
	plt.plot(a,b)

	plt.show()


write('test.wav', 6000, b)

