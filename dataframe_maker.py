#%%
import pandas as pd
import tweepy as tw
from datetime import datetime
from datetime import timedelta
import numpy as np
from emosent import get_emoji_sentiment_rank
import json
from Named_Entity import SpaCy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

class dataframe:
    
    def __init__(self):
        self.SEARCH_TYPE='mixed'    #what type of search you want: popular/mixed/recent
        self.ITEMS_TO_QUERY=500       #number of tweets to query per day
        self.DATE=datetime.date(datetime.now())    #today's date
        self.TIMEDELTA=7            # how much into the past you want to go into ( in days )
        self.THRESHOLD=20
        with open(r'C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\Named ER\Adverse_terms.json') as f:    
            self.adverse_word=json.load(f)
            f.close()
    
    
    def emoji_sentiment(self,text):
    

        # this function is there to analyse and return the emoji sentiment per text in a list of emoji-filled text docs
        emoji=[] #emoji is a list of all the emoji sentiments found in a list of documents
        #avg_emoji_sentiment=0.0 #avg_ emoji sent is a list of the average sentiment defined by emojis in a list of docs

        

        doc=text.split()
        avg_emoji_sentiment=0.0  # we use 0.0 as the neutral sentiment here and if the doc does not contain any emoji or 
                                        #if the emoji is not recognised by the function then 0.0 gets appended to the 'emoji' list
        emoji.append([])

        for word in doc:
            try:
                emoji[-1].append(get_emoji_sentiment_rank(word))
            except:
                pass

        if emoji[-1]:
            avg_emoji_sentiment = round(np.mean(emoji),3)
        #returns a list of the avg emoji sentiment per document in a list of docs
        return avg_emoji_sentiment

    
    
    
    def adverse_words(self,text,adverse_word): # checks if any adverse word is present in our tweet
        adverse_list=[]
        text=text.split()
        for word in text:
            if word in adverse_word:
                adverse_list.append(word)
   
        return adverse_list  

    def negative_col(self,lst): # if the text contains a word from the given adverse words list then it returns 1, else 0
        if len(lst)==0:
            return 0
        else:
            return 1
    
    
    def sentiment_analyzer_scores(self,sentence):
        sia=SIA()
        score = sia.polarity_scores(sentence)
        #print("{} {}".format(sentence, score))
        return score
    

    def query_tweets(self,SEARCH_WORDS,DATE_TO_SEARCH,ITEMS_TO_QUERY):   

        consumer_key='XrJHIZzeveP06K992nXShtUXS'
        consumer_secret='kZl0F2ghB8ecqI1PK6NjV5tKW4szTNW5bnyDmeuCKHF6Gbdttd'
        access_token_key='1295434260514402307-pq7jzeDl1j8piM3Bpqo1Y1HoFXRCEQ'
        access_token_secret='Gqks3ZhgD45YTYAkaqLQBUjblkPfGDjhJbgjnXG3jUz0r'

        auth = tw.OAuthHandler(consumer_key, consumer_secret)  #authentications for accessing the twitter api
        auth.set_access_token(access_token_key, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)

        SEARCH_TYPE=self.SEARCH_TYPE
        # Collect tweets
        # We can only query only upto 7 days of tweets at one time
        tweets = tw.Cursor(api.search,
                    q=SEARCH_WORDS,
                    lang="en",
                    until=DATE_TO_SEARCH,result_type=SEARCH_TYPE
                    ,tweet_mode='extended').items(ITEMS_TO_QUERY)
        
        
        twit=list()
        retweets_count=list()  #how many retweets the tweet has had
        retweet=list()         #if the tweet has been retweeted or not
        thresh=list()          #if the tweet can be classified as popular or not
        date=list()            #date of the tweet
        adverse_list=list()    #what adverse words are present in the tweet
        adverse_presence=list() #if adverse words are present in the tweet
        name=list()            #name of user
        location=list()        #location of user
        url=list()             #url of tweet
        verified=list()        # if the user is verified or not
        hashtags=list()        #what hashtags are contained in the tweet
        emoji=list()           #the emoji sentiment for the tweet 0.0 being neutral
        screen_name=list()     #screen name of user of the tweet
        
        for tweet in tweets:
            
            tweet.full_text=tweet.full_text.split(sep='https://t.co/')[0]  #removing tweet url from tweet
            if tweet.full_text in twit:  #sometimes we get repeated tweets
                continue

            if tweet.full_text[:2]=='RT':
                continue

            
            #retweeted, possibly_sensitive #retweeted #entities
            emoji.append(self.emoji_sentiment(tweet.full_text))
            
            
            if len(tweet.entities['hashtags'])==0:    #sometimes no tweet is present so its an empty list
                hashtags.append(None)
            else:
                hashtags.append(tweet.entities['hashtags'][0]['text'])
            
            if len(tweet.entities['urls'])==0:        #sometimes no url is present so empty list
                url.append(None)
            else:
                url.append(tweet.entities['urls'][0]['url'])
            verified.append(tweet.user._json['verified'])
            location.append(tweet.user._json['location'])
            screen_name.append(tweet.user._json['screen_name'])
            name.append(tweet.user._json['name'])
            twit.append(tweet.full_text)
            retweets_count.append(tweet.retweet_count)
            date.append(datetime.date(tweet.created_at))
            adverse_list.append(self.adverse_words(tweet.full_text,self.adverse_word))
            adverse_presence.append(self.negative_col(adverse_list[-1]))
            
            

            if tweet.retweet_count>0:
                retweet.append(1)
            else:
                retweet.append(0)
            if tweet.retweet_count>=self.THRESHOLD:
                thresh.append(1)
            else:
                thresh.append(0)


        thresh=pd.Series(thresh)
        twit=pd.Series(twit)
        retweets_count=pd.Series(retweets_count)
        retweet=pd.Series(retweet)
        date=pd.to_datetime(pd.Series(date,name='Date'))
        adverse_list=pd.Series(adverse_list)
        adverse_presence=pd.Series(adverse_presence)
        name=pd.Series(name)
        location=pd.Series(location)
        verified=pd.Series(verified)
        hashtags=pd.Series(hashtags)
        url=pd.Series(url)
        emoji_sent=pd.Series(emoji)
        screen_name=pd.Series(screen_name)
        
        

        df=pd.concat([name,twit,screen_name,location,hashtags,url,retweets_count,retweet,thresh,verified,adverse_list,adverse_presence,emoji_sent],axis=1)
        df.set_index(date,inplace=True)

        df.columns=['Name','Tweet','Screen_Name','Location','Hashtags','Url','Retweet_count','Retweet','Popular','Verified','Adverse_terms','Adverse_presence','Emoji_Sentiment']
        s=SpaCy()
        df['Entities']=s.NER(df.Tweet)
        df['Organisation']=df.Entities.apply(lambda x: s.ORG(x))
        df['VADER']=df.Tweet.apply(lambda x: self.sentiment_analyzer_scores(x)['compound'])
        
        return df
 
    
    
    def dataframe_generator(self,SEARCH_WORDS,ITEMS_TO_QUERY=None,TIMEDELTA=None):
        
        lst=[]
        if ITEMS_TO_QUERY==None:             #if no arguement is provided for how many items to query per day, use the default
            ITEMS_TO_QUERY=self.ITEMS_TO_QUERY
            
        if TIMEDELTA==None:                  #if no arguement is provided for how many days to look back, use default
            TIMEDELTA=self.TIMEDELTA

            
        DATE=self.DATE   
        
        err=0
        while DATE>datetime.date(datetime.now()-timedelta(TIMEDELTA)):
            if err<TIMEDELTA+1:
                DATE=str(DATE)
                df=self.query_tweets(SEARCH_WORDS,DATE_TO_SEARCH=DATE,ITEMS_TO_QUERY=ITEMS_TO_QUERY)
                lst.append(df)
                #DATE=datetime.date(df.index[-1])
                DATE=datetime.date(datetime.strptime(DATE, '%Y-%m-%d'))-timedelta(1)
                err+=1
            else: 
                raise RuntimeError("Your loop has run more than the anticipated value")
                

        data=pd.concat([df for df in lst])
        return data


# %%
df=dataframe()
data=df.dataframe_generator("PS5",ITEMS_TO_QUERY=10,TIMEDELTA=1)

# %%
data
#find out the most popular words used in our dataframe
#finding out the most commmon topic in our tweets - topic detection

#from: user 
#@: check for user mention
# lang: checks the language of the tweet
# #: specific hashtag presence 
# OR: joins multiple requests together


# %%
