def f(x):
	return x**3 + 10

# sol 5304

def RHRS(f = f, a=0, b=12, n=1000000):
	dX = (b-a)/float(n)
	area = 0
	for i in range(0, n):
		a += dX
		fx = f(a)
		area += fx*dX
	return area

a = []
b = []
c = 5
for i in range(0, 12):
	c *= 2
	b.append(c)
print b
for i in b:
	a.append(RHRS(n = i))

print a
import matplotlib.pyplot as plt

plt.plot(b,a)
plt.show()


