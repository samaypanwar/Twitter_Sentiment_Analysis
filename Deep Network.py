# %%
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
df=pd.read_csv(r'C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\Lemmatized_text')

# %%
df.dropna(inplace=True)
df = shuffle(df)
df = df.sample(frac=1).reset_index(drop=True)
labels=df['Polarity']
df.drop(['Unnamed: 0','Polarity'],axis=1,inplace=True)

# %%
for i in range(len(labels)):
    if labels[i]==4:
        labels[i]=1
print(labels)

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,SpatialDropout1D,LSTM,Dense,Flatten
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split

# %%
tweet=df.lemma.values
label=labels.values

token=Tokenizer(lower=False)
token.fit_on_texts(tweet)
max_len=37
train_tokenized=token.texts_to_sequences(tweet)
X=pad_sequences(train_tokenized,maxlen=max_len)

x_train,x_test,y_train,y_test=train_test_split(X,label,test_size=0.2,shuffle=True)
#while model_class_imbalance(y_train,threshold=0.05) or model_class_imbalance(y_test,0.05)==True:
    #x_train,x_test,y_train,y_test=train_test_split(X,label,test_size=0.2,shuffle=True)

print(" shape of x train and y train: ",x_train.shape,y_train.shape)
print(" shape of x test and y test: ",x_test.shape,y_test.shape)




# %%
import tensorflow 
from sklearn.metrics import accuracy_score
from tqdm import tqdm
embed=150


lstm_units=50
vocab_size=X.max()+1


acc=0      
model=tensorflow.keras.Sequential([
Embedding(vocab_size,embed,input_length=X.shape[1]),
SpatialDropout1D(0.2),
LSTM(lstm_units,dropout=0.5,recurrent_dropout=0.5),
Dense(1,activation='sigmoid')           
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
history = model.fit(x_train,y_train,validation_split=0.2, epochs=3, batch_size=64)

ypred=model.predict(x_test)
for i in range(len(ypred)):
    if ypred[i]>=0.5:
        ypred[i]=1
    else:
            ypred[i]=0
score=accuracy_score(y_test,ypred)
if score>=acc:
    acc=score
    



#%%

#print("The configurations of your model should be: {:d} embedding dimensions, {:d} batch size".format(config[0],config[1]) )
print('Accuracy of model: ',acc)
# %%
import matplotlib.pyplot as plt

plt.plot(history.history['loss'],label='Loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend()
plt.grid(True)
plt.show()
# %%
model.evaluate(x_test,y_test)
# %%
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.legend()
plt.grid(True)
plt.show()
