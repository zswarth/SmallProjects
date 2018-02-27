from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer

class Clusterer(object):
  sample_text = """
    Text is made of characters, but files are made of bytes. In Unicode, there are many more possible characters than possible bytes. Every text file is encoded so that its characters can be represented as bytes. When you work with text in Python, it should be Unicode. Most of the text feature extractors in scikit-learn will only work with Unicode. So to correctly load text from a file (or from the network), you need to decode it with the correct encoding. An encoding can also be called a charset or character set, though this terminology is less accurate. The CountVectorizer takes a encoding parameter to tell it what encoding to decode text from. For modern text files, the correct encoding is probably UTF-8. The CountVectorizer has encoding=utf-8 as the default. If the text you are loading is not actually encoded with UTF-8, however, you will get a UnicodeDecodeError.
  """

  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.clf = None

  def cluster(self, text = None):
    if not text:
      text = self.__class__.sample_text.split('.')
    cv = CountVectorizer()
    self.X = cv.fit_transform(text)
    self.clustered = self.clf.fit_predict(self.X)
    self.all_text = [[] for x in xrange(self.n_clusters)]
    for idx, cluster in enumerate(self.clustered):
      self.all_text[cluster].append(text[idx])

    return self.all_text

  def __repr__(self):
    return "Some string"
    
class KMeansClusterer(Clusterer):

  def __init__(self, n_clusters = 3):
    super(KMeansClusterer, self).__init__(n_clusters)
    self.clf = KMeans(n_clusters = self.n_clusters)


class SpectralClusteringCluster(Clusterer):

  def __init__(self, n_clusters = 3):
    super(SpectralClusteringCluster, self).__init__(n_clusters)
    self.clf = SpectralClustering(n_clusters = self.n_clusters)




# def foo(a = 'a', b = 'b', **kwargs):
#  print a
#  print b
#  for kw, arg in kwargs.iteritems():
#    print kw, arg

#def foo2(a = 'a', b = 'b'):
#  print a
#  print b

#def foo3(a, b):
#  print a
#  print b

#def foo4(*args):
  #for arg in args:
 #   print arg

#foo(d = 'something')

# kwargs = {'d': 'something'}

# kwargs['d']