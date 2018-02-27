
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import sklearn.cluster as sk


img=mpimg.imread('test.png')

clf = sk.KMeans(n_clusters = 4)

a,b,c = np.shape(img)
print a,b

img2 = img.reshape(a*b, 3)

clf.fit(img2)

img3 = np.zeros((a,b, 3))
center = clf.cluster_centers_

for i in range(0,a):
    for j in range(0,b):
        img3[i,j] = center[clf.predict([img[i,j]])]

imgplot = plt.imshow(img)
imgplot = plt.imshow(img3)

