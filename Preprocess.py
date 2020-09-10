#%%
import numpy as np
import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
class Preprocessing:
     
    def __init__(self,df):
        self.df=df
    
        
    def class_imbalance(self,dataset,threshold,target='Polarity',num_classes=2):
        c=[]
        l=dataset[target].count()/num_classes
        print("Ideal class size : ",l)
        for i in range(num_classes):
            k=np.array(dataset[dataset[target]==4*i])

            c.append(len(k))
            print("Length of class",i,": ",c[i])
            if (1-threshold)*l<c[i]<(1+threshold)*l:
                continue

            else:
                print('There exists a Class Imbalance in your Dataset')
                return True
                
        return False

    

    class PreProcessTweets:

        def pre_processTweet(self,tweet):
            tweet=tweet.lower()  #to convert the text to lower case
            tweet=re.sub('((www.[^s]+)|(https?://[^s]+))','URL',tweet) #to remove URLs
            tweet=re.sub('@[^s]+','AT_USER',tweet) #to remove usernames
            tweet=re.sub('#([^s]+)','1',tweet)# to remove the hashtag


            return tweet    

        def processTweets(self,list_of_tweets):
            processedTweets=[]
            for tweet in list_of_tweets:
                tweet = self.pre_processTweet(tweet)
                processedTweets.append(tweet)
            return processedTweets

    def clean_text(self,x):
    
        tw=['AT_USER','URL']
        text=[word for word in x.split() if word not in tw ]
        text=" ".join(text)
        return text
    
    
    def clean(self,tweet):
        tweet = re.sub(r"he's", "he is", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"We're", "We are", tweet)
        tweet = re.sub(r"That's", "That is", tweet)
        tweet = re.sub(r"won't", "will not", tweet)
        tweet = re.sub(r"they're", "they are", tweet)
        tweet = re.sub(r"Can't", "Cannot", tweet)
        tweet = re.sub(r"wasn't", "was not", tweet)
        tweet = re.sub(r"aren't", "are not", tweet)
        tweet = re.sub(r"isn't", "is not", tweet)
        tweet = re.sub(r"What's", "What is", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"should've", "should have", tweet)
        tweet = re.sub(r"where's", "where is", tweet)
        tweet = re.sub(r"we'd", "we would", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"weren't", "were not", tweet)
        tweet = re.sub(r"They're", "They are", tweet)
        tweet = re.sub(r"let's", "let us", tweet)
        tweet = re.sub(r"it's", "it is", tweet)
        tweet = re.sub(r"can't", "cannot", tweet)
        tweet = re.sub(r"don't", "do not", tweet)
        tweet = re.sub(r"you're", "you are", tweet)
        tweet = re.sub(r"i've", "I have", tweet)
        tweet = re.sub(r"that's", "that is", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"doesn't", "does not", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"didn't", "did not", tweet)
        tweet = re.sub(r"ain't", "am not", tweet)
        tweet = re.sub(r"you'll", "you will", tweet)
        tweet = re.sub(r"I've", "I have", tweet)
        tweet = re.sub(r"Don't", "do not", tweet)
        tweet = re.sub(r"I'll", "I will", tweet)
        tweet = re.sub(r"I'd", "I would", tweet)
        tweet = re.sub(r"Let's", "Let us", tweet)
        tweet = re.sub(r"you'd", "You would", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"Ain't", "am not", tweet)
        tweet = re.sub(r"Haven't", "Have not", tweet)
        tweet = re.sub(r"Could've", "Could have", tweet)
        tweet = re.sub(r"youve", "you have", tweet)
        tweet = re.sub(r"haven't", "have not", tweet)
        tweet = re.sub(r"hasn't", "has not", tweet)
        tweet = re.sub(r"There's", "There is", tweet)
        tweet = re.sub(r"He's", "He is", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"You're", "You are", tweet)
        tweet = re.sub(r"I'M", "I am", tweet)
        tweet = re.sub(r"shouldn't", "should not", tweet)
        tweet = re.sub(r"wouldn't", "would not", tweet)
        tweet = re.sub(r"i'm", "I am", tweet)
        tweet = re.sub(r"I'm", "I am", tweet)
        tweet = re.sub(r"Isn't", "is not", tweet)
        tweet = re.sub(r"Here's", "Here is", tweet)
        tweet = re.sub(r"you've", "you have", tweet)
        tweet = re.sub(r"we're", "we are", tweet)
        tweet = re.sub(r"what's", "what is", tweet)
        tweet = re.sub(r"couldn't", "could not", tweet)
        tweet = re.sub(r"we've", "we have", tweet)
        tweet = re.sub(r"who's", "who is", tweet)
        tweet = re.sub(r"y'all", "you all", tweet)
        tweet = re.sub(r"would've", "would have", tweet)
        tweet = re.sub(r"it'll", "it will", tweet)
        tweet = re.sub(r"we'll", "we will", tweet)
        tweet = re.sub(r"We've", "We have", tweet)
        tweet = re.sub(r"he'll", "he will", tweet)
        tweet = re.sub(r"Y'all", "You all", tweet)
        tweet = re.sub(r"Weren't", "Were not", tweet)
        tweet = re.sub(r"Didn't", "Did not", tweet)
        tweet = re.sub(r"they'll", "they will", tweet)
        tweet = re.sub(r"they'd", "they would", tweet)
        tweet = re.sub(r"DON'T", "DO NOT", tweet)
        tweet = re.sub(r"they've", "they have", tweet)
        return tweet



    def get_wordnet_pos(self,tag):
        
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    
    def lemmatize(self,text):
        from string import punctuation
        stopword=set(stopwords.words('english'))
        stopword=stopword.union(set(punctuation))
        
        lemmatizer=WordNetLemmatizer()

        text=word_tokenize(text)
        tagged=nltk.pos_tag(text)
        string=[]

        for index,word in enumerate(text):
            if word not in stopword:
                word=lemmatizer.lemmatize(word,pos=self.get_wordnet_pos(tagged[index][1]))
                string.append(word)
        string=" ".join(string)
        return string
    
    def __call__(self):

        preprocess=self.PreProcessTweets()

        self.df['Tweet']=preprocess.processTweets(df['Tweet'])
        self.df['Tweet']=df['Tweet'].apply(lambda x: self.clean_text(x))
        self.df['Tweet']=df['Tweet'].apply(lambda x: self.clean(x))
        self.df['Tweet']=df['Tweet'].apply(lambda x: self.lemmatize(x))

        df.dropna(inplace=True)

        tweet=df.Tweet.values
        label=df.Polarity.values
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(tweet)
        MAX_SEQUENCE_LENGTH=37
        sequences=tokenizer.texts_to_sequences(tweet)
        word_index=tokenizer.word_index
        data=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

        return (data,label)

# %%
import pandas as pd
from sklearn.utils import shuffle
import re
df=pd.read_csv(r"C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\dataset\test_data_full.csv",header=None)
df.dropna(inplace=True)
df = shuffle(df)
df = df.sample(frac=1).reset_index(drop=True)
df.columns=['Polarity','ID','Date','Query','USER_ID','Tweet']
drop=['ID', 'Date', 'Query', 'USER_ID']
df.drop(drop,axis=1,inplace=True)
#labels=df['Polarity']
df.head()

# %%
pre=Preprocessing(df)
data,label=pre()
# %%


