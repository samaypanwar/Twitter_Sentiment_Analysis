import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class PreProcessTweets:
    
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])

    def pre_processTweet(self,tweet):
        tweet=tweet.lower()  #to convert the text to lower case
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet) #to remove URLs
        tweet=re.sub('@[^\s]+','AT_USER',tweet) #to remove usernames
        tweet=re.sub('#([^\s]+)','\1',tweet)# to remove the hashtag
        tweet=word_tokenize(tweet)
        return tweet    
        
    def processTweets(self,list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            tweet = self.pre_processTweet(tweet)
            processedTweets.append(tweet)
        return processedTweets