from fractions import gcd
import random


#create RSA
def find_coprime(b):
	coprimes = []
	for i in range(2, b):
		if gcd(i, b) == 1:
			coprimes.append(i)
	a = len(coprimes)
	b = random.randrange(0,a)
	return coprimes[b]


def find_private(e, tol):
	possable = []
	for d in range(1,1000):
		if (d*e)%tol == 1:
			possable.append(d)
	a = len(possable)
	b = random.randrange(0,a)
	return possable[b]

def RSA_Create(p, q):
	n = p*q
	tol = (p-1)*(q-1)
	e = find_coprime(tol)
	d = find_private(e,tol)
	return n,e,d

n = 391
e = 245
d = 125

def RSA_incode(message, n = 391, e = 245):
	E = (message**e)%n
	return E


def RSA_decode(E, d = 125, n = 391):
	message = (E**d)%n
	return message



encoded = RSA_incode(18)
print encoded
print RSA_decode(encoded)

	