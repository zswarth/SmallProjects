import tweetpony
import re

class Twitter(object):

  access_token = '2376312126-YUCVGSOYgP9kQJNx46AOwIRCVddKUPpcf7fDkJH'
  access_token_secret = '8ugXhMtCYq0IHP5KAzM16xoiMvXkXIpWtTWvk52V9QCbS'
  consumer_key = '7gNH5uIskTOTACpCOKgbw'
  consumer_secret = '4wiDP0muI5Krudzt8cVT30w1QKqMKjqjLGYTobbKY'

  def __init__(self):
    self.reset_auth_token()
    self.try_later = []
    self.num_processed = 0
    self.stop_words = self.get_stopwords()

  def reset_auth_token(self):
    self.api = tweetpony.API(
        consumer_secret = self.__class__.consumer_secret,
        consumer_key = self.__class__.consumer_key,
        access_token = self.__class__.access_token,
        access_token_secret = self.__class__.access_token_secret
    )

  def get_stopwords(self):
    f = open('/Users/zacharyswarth/Desktop/stop_words.txt')
    words = []
    words2 = []
    k = []
    for line in f:
      words.append([line])
    for i in words:
      k.append(i[0])
    for j in k:
      b = re.search(r'\w+', j)
      c = b.group()
      words2.append(c)
    return words2

  def tweet_text(self, query, n_tweets = 5):
    tweets = []
    tweets2 = []
    tweets3 = []
    a = self.api.search_tweets(q= query, count = n_tweets)
    b = a['statuses']
    for i in b:
     tweets.append(i['text'])
    for i in tweets:
      a = re.sub('#', '', i)
      b = re.sub('@\w+', '', a)
      c = re.sub('http:[\w.-/]+', '', b)
      tweets2.append(c)
    for i in tweets2:
      words = i.split()
      for i in words:
        if i in self.stop_words:
          words.remove(i)
      tweets3.append(' '.join(words))

    return tweets3

  def tweets(self, query, n_tweets = 5):
    tweets = []
    a =self.api.search_tweets(q=query, count = n_tweets)
    b = a['statuses']
    for i in b:
      tweets.append(i['text'])
    return tweets

  @property
  def rate_limit_exceeded(self):
    return self.num_processed >= 175 


