#%%
import requests
import os
import json
import math

class Call_Tweets:

    def __init__(self,ITEMS_TO_QUERY=None,query=None,tweet_fields=None,user_fields=None):
        """
        ITEMS_TO_QUERY: total number of tweets you want, must be greater than max_results"
        query: the query you want to search for
        tweet_fields: the fields you want included in your response object
        user_field: the user fields you want included in your response object
        """
        self.BEARER_TOKEN='AAAAAAAAAAAAAAAAAAAAAIV4GwEAAAAAtU8I9t4ZPNU2d%2FlJfLwrvdaqvXs%3D4Gz5JpTs843syzkgK1hVYuHs0nOxAP8xc1H6JgmDJ7vpaKOIDT'
        self.query=query
        self.tweet_fields=tweet_fields
        self.user_fields=user_fields
        self.ITEMS_TO_QUERY=ITEMS_TO_QUERY  
        self.NEXT_TOKEN=None       # to help twitter scroll the pages and give results further back
        
        self.MAX_RESULTS=10

    def create_url(self):
        query = self.query
        # Tweet fields are adjustable.
        # Options include:
        # attachments, author_id, context_annotations,
        # conversation_id, created_at, entities, geo, id,
        # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
        # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
        # source, text, and withheld
        tweet_fields = "tweet.fields="+self.tweet_fields+',lang' #"tweet.fields=created_at,entities"
        user_fields='user.fields='+self.user_fields #'user.fields=username,verified'
        tuple_query=[query,tweet_fields,user_fields]
        max_results='max_results='+str(self.MAX_RESULTS)
        
        if self.NEXT_TOKEN==None:     # if this is the first call then there is no next_token object
        
            url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&{}&{}".format(
                tuple_query[0],tuple_query[1],tuple_query[2],max_results
            )
        else:
            next_token='next_token='+self.NEXT_TOKEN
            url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&{}&{}&{}".format(
                tuple_query[0],tuple_query[1],tuple_query[2],next_token,max_results
            )
        # print(url)
        return url


    def create_headers(self,bearer_token):
        """
        Bearer token: the token of authorisation provided for your application by twitter
        """
        headers = {"Authorization": "Bearer {}".format(bearer_token)}
        return headers


    def connect_to_endpoint(self,url,headers):

        response = requests.request("GET", url, headers=headers)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()

    def getTweets(self):

        bearer_token =self.BEARER_TOKEN
        url = self.create_url()
        headers = self.create_headers(bearer_token)
        json_response = self.connect_to_endpoint(url, headers)
        
        return json_response

    def __call__(self):

        ITEMS_TO_QUERY=self.ITEMS_TO_QUERY
        data=[]
  
        api_calls=math.ceil(ITEMS_TO_QUERY/self.MAX_RESULTS)
        for call in range(api_calls):
            data.append(self.getTweets())
            self.NEXT_TOKEN=data[call]['meta']['next_token']

        return data
    
def tweet_query(OR=[],AND=[],NOT=[],lang='en',retweet=False,**kwargs):

    """
    OR: the words you want in a list form
    AND: all these words should be present in response tweet, should be in list form
    lang: default is english
    retweet: default is that it should not be a retweet

    """
    if retweet==False:
        retweet=' -is:retweet'
    else:
        retweet=' is:retweet'
    
    query_lang=' lang:'+lang
    query_or=''
    query_and=''
    query_not=''
 
    for word in OR:         
        query_or+=' {} OR'.format(word)     
    query_or=query_or[:-4]
    
    if query_or=='':
        pass
    else:
        query_or=' ({})'.format(query_or)

    for word in AND:
        query_and+=' {}'.format(word)

    for word in NOT:
        query_not+=' -{}'.format(word)
    
    query=query_and+(query_or)+query_not+retweet+query_lang
    
    
    return query

#%%
query=tweet_query(AND=['CLOSE','PRICE'],OR=['AAPL','MSFT'])
#%%
j=Call_Tweets(ITEMS_TO_QUERY=1,query=query,tweet_fields='created_at,entities,public_metrics,possibly_sensitive,referenced_tweets,source,text',user_fields='verified')()

# %%
"""
https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent

https://developer.twitter.com/en/docs/twitter-api/v1/rules-and-filtering/overview/standard-operators


"""
