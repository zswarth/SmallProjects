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


  def reset_auth_token(self):
    self.api = tweetpony.API(
        consumer_secret = self.__class__.consumer_secret,
        consumer_key = self.__class__.consumer_key,
        access_token = self.__class__.access_token,
        access_token_secret = self.__class__.access_token_secret
    )


  def tweets(self, query, n_tweets = 50):
    tweets = []
    tweets2 = []
    tweets3 = []
    a =self.api.search_tweets(q=query, count = n_tweets)
    b = a['statuses']
    for i in b:
      tweets.append(i['text'])
    for i in tweets:
      a = re.sub('#', '', i)
      b = re.sub('@\w+', '', a)
      c = re.sub('RT :', '', b)
      d = re.sub('http:\w+', '', c)
      tweets2.append(d)
    for i in tweets2:
      a = i.encode('UTF-8')
      tweets3.append(a)
    return tweets3


  def tweets_empty(self, tweets):
    tweets2= []
    for i in tweets:
      a = re.sub('#', '', i)
      b = re.sub('u', '', a)
      tweets2.append(b)
    return tweets2

  

  @property
  def rate_limit_exceeded(self):
    return self.num_processed >= 175 


import sys
import tweepy
 
#values to search for - more can be added
values = ['red','blue']
 
# Get these values from your dev.twitter application settings.
access_token = '2376312126-YUCVGSOYgP9kQJNx46AOwIRCVddKUPpcf7fDkJH'
access_token_secret = '8ugXhMtCYq0IHP5KAzM16xoiMvXkXIpWtTWvk52V9QCbS'
consumer_key = '7gNH5uIskTOTACpCOKgbw'
consumer_secret = '4wiDP0muI5Krudzt8cVT30w1QKqMKjqjLGYTobbKY'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
#open file to store tweets - specify with OS path i.e. windows could be c:\workfile for example
f = open('test.tx', 'w')
 
 
class CustomStreamListener(tweepy.StreamListener):
        def on_status(self, status):
               
                try:
                        # Write new line and tweet status
                        f.write ('\n')
                        original_tweet = (status.text + '\n')
                        encoded_tweet = original_tweet.encode('UTF-8')
                        f.write (encoded_tweet)
 
                        # Write new line and tweet time                
                        original_tweet_time = (str(status.created_at) + '\n')
                        encoded_tweet_time = original_tweet_time.encode('UTF-8')
                        f.write (encoded_tweet_time)
 
                        # Write new line and author of tweet
                        original_name = (str(status.author.screen_name) + '\n')
                        encoded_name = original_name.encode('UTF-8')
                        f.write (encoded_name)
 
                except Exception, e:
                        print >> sys.stderr, 'Encountered Exception:', e
                        pass
 
        def on_error(self, status_code):
                print >> sys.stderr, 'Encountered error with status code:', status_code
                return True # Leave stream running
       
        def on_timeout(self):
                print >> sys.stderr, 'Timeout...'
                return True # Leave stream running
 
 
 
def main():
        # Streaming API and set a timeout value of 60 seconds.
        streaming_api = tweepy.streaming.Stream(auth, CustomStreamListener(), timeout=60)
 
        print >> sys.stderr, 'Filtering the public timeline for ', values
        streaming_api.filter(track=values)
 
if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        f.close()
        print "See ya!"