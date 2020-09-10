# %% 
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
# %%
df=pd.read_csv(r"C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\dataset\train_data_full.csv",
encoding="ISO-8859-1",header=None)
df = shuffle(df)

df = df.sample(frac=0.00625).reset_index(drop=True)
# %%
df.columns=['Polarity','ID','Date','Query','USER_ID','Tweet']
drop=['ID', 'Date', 'Query', 'USER_ID']
df.drop(drop,axis=1,inplace=True)
df.head()
#%%
for i in range(len(df)):
    if df['Polarity'][i]==4:
        df['Polarity'][i]=1
print(df['Polarity'])
# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
sia=SIA()
def sentiment_analyzer_scores(sentence):
    score = sia.polarity_scores(sentence)
    #print("{} {}".format(sentence, score))
    return score
# %%
df['VADER']=df['Tweet'].apply(lambda x: sentiment_analyzer_scores(x))
'''for i in range(len(df)):
    if df['VADER'][i]['compound']>=0:    
        df['Predicted Sentiment'][i]=1
    else:
        df['Predicted Sentiment'][i]=0

from sklearn.metrics import accuracy_score
score=accuracy_score(df['Polarity'].values,df['Predicted Sentiment'].values)
print('VADER has achieved an accuracy score of: ',score)'''

 
# %%
df.head()
# %%
