# %%
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
df=pd.read_csv(r'C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\Hyperparameter_tuning_data.csv')

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
tweet=df.cleaned.values
label=labels.values

tokenizer=Tokenizer()
tokenizer.fit_on_texts(tweet)
MAX_SEQUENCE_LENGTH=37
sequences=tokenizer.texts_to_sequences(tweet)
word_index=tokenizer.word_index
data=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2,shuffle=True)
#while model_class_imbalance(y_train,threshold=0.05) or model_class_imbalance(y_test,0.05)==True:
    #x_train,x_test,y_train,y_test=train_test_split(X,label,test_size=0.2,shuffle=True)

print(" shape of x train and y train: ",x_train.shape,y_train.shape)
print(" shape of x test and y test: ",x_test.shape,y_test.shape)


#%%
import os
import numpy as np

embeddings_index = {}
GLOVE_DIR='INTERN'
f = open('glove.twitter.27B.25d.txt',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#%%
EMBEDDING_DIM=25
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('done')

# %%
import tensorflow 
from sklearn.metrics import accuracy_score
from tqdm import tqdm



lstm_units=[16]#,32,64,128,256]
batch_size=[32]#,64,128,256,512,1024]
dropout=[0.1]#,0.2,0.3,0.4,0.5]
optimizer=['adam']#,'adamax','sgd']
rec_dropout=[0.1,0.2,0.3,0.4,0.5]

model_spec=[]
acc=0
for lstm_unit in lstm_units:
    for batch in batch_size:
        for drop in dropout:
            for op in optimizer:
                for rec_drop in rec_dropout:    
                    model=tensorflow.keras.Sequential([
                    
                    Embedding(len(word_index) + 1,
                                                EMBEDDING_DIM,
                                                weights=[embedding_matrix],
                                                input_length=MAX_SEQUENCE_LENGTH,
                                                trainable=False),
                    SpatialDropout1D(drop),
                    LSTM(lstm_unit,dropout=0.5,recurrent_dropout=rec_drop),
                    Dense(1,activation='sigmoid')           
                    ])
                    model.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
                    model.summary()
                    history = model.fit(x_train,y_train,validation_split=0.3, epochs=5,batch_size=batch )

                    out=model.evaluate(x_test,y_test)
                    if out[1]>acc:
                        acc=out[1]
                        spec=(lstm_unit,batch,drop,op,rec_drop)
                        model_spec.append(spec)

                   
    



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