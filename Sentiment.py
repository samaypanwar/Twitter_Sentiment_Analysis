# %% 
class Model():

    def tfidf_embedding(self,lemmatized_dataframe):
        from nltk.tokenize import RegexpTokenizer
        from sklearn.feature_extraction.text import TfidfVectorizer


        token=RegexpTokenizer(r'[a-zA-Z0-9]+')
        tfidf=TfidfVectorizer(stop_words='english',tokenizer=token.tokenize)
        text=tfidf.fit_transform(lemmatized_dataframe)

        return text
    
 
    def model_class_imbalance(self,label_dataframe,threshold,num_classes=2):

        c=[]
        l=len(label_dataframe)/num_classes

        print("**********************************")
        print("Ideal class size : ",l)

        for i in range(num_classes):

            k=label_dataframe[label_dataframe==4*i].count()
            c.append(k)

            print("Length of class",i,": ",c[i])
            if (1-threshold)*l<c[i]<(1+threshold)*l:
                continue

            else:
                print('There exists a Class Imbalance in your Dataset')
                return True
        return False
        

    def test_train(self,vectorized_matrix,label_dataframe):

        from sklearn.model_selection import train_test_split

        
        x_train,x_test,y_train,y_test=train_test_split(vectorized_matrix,label_dataframe,test_size=0.2,shuffle=True)
        while self.model_class_imbalance(y_train,threshold=0.05)==True or self.model_class_imbalance(y_test,0.05)==True:
            x_train,x_test,y_train,y_test=train_test_split(self,vectorized_matrix,label_dataframe,test_size=0.2,shuffle=True)

        return x_train,x_test,y_train,y_test
    

    def naive_bayes(self,vectorized_matrix,label_dataframe):

        from sklearn import metrics
        from sklearn.naive_bayes import MultinomialNB,BernoulliNB

        def multinomialNB(self,vectorized_matrix,label_dataframe):

            x_train,x_test,y_train,y_test=self.test_train(vectorized_matrix,label_dataframe)
            clf1=MultinomialNB()
            clf1.fit(x_train,y_train)
            ypred_MNB=clf1.predict(x_test)
            print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, ypred_MNB))

            return ypred_MNB
            
        def bernoulliNB(self,vectorized_matrix,label_dataframe):

            x_train,x_test,y_train,y_test=self.test_train(vectorized_matrix,label_dataframe)
            clf3 = BernoulliNB().fit(x_train,y_train)
            ypred_BNB= clf3.predict(x_test)
            print("BernoulliNB Accuracy:",metrics.accuracy_score(y_test,ypred_BNB))

            return ypred_BNB

          
    def Deep_network(self,lemmatized_dataframe_w_labels): 
        import pandas as pd
        import numpy as np
        from sklearn.utils import shuffle

        #df=pd.read_csv(r'C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\Lemmatized_text')

        df=lemmatized_dataframe_w_labels.copy()
        df.dropna(inplace=True)
        df = shuffle(df)
        df = df.sample(frac=1).reset_index(drop=True)
        label=df.Polarity.values
        df.drop(['Unnamed: 0','Polarity'],axis=1,inplace=True)

        
        for i in range(len(label)):
            if label[i]==4:
                label[i]=1
        print(label)

        
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.layers import Embedding,SpatialDropout1D,LSTM,Dense,Flatten
        from sklearn.model_selection import train_test_split
        from tensorflow.keras import Sequential

        
        tweet=df.lemma.values
        token=Tokenizer(lower=False)
        token.fit_on_texts(tweet)
        max_len=37
        train_tokenized=token.texts_to_sequences(tweet)
        X=pad_sequences(train_tokenized,maxlen=max_len)

        x_train,x_test,y_train,y_test=train_test_split(X,label,test_size=0.2,shuffle=True)
        
        #cannot use test_train function as that requires the data to be in Pandas Dataframe format

        print("shape of x train and y train: ",x_train.shape,y_train.shape)
        print("shape of x test and y test: ",x_test.shape,y_test.shape)


        #hyperparameter settings
        embed_dim=100
        lstm_units=50
        vocab_size=X.max()+1

        model=Sequential([

            Embedding(vocab_size,embed_dim,input_length=X.shape[1]),
            SpatialDropout1D(0.25),
            LSTM(lstm_units,dropout=0.5,recurrent_dropout=0.5),
            Dense(1,activation='sigmoid') 

        ])

        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()

        history = model.fit(x_train,y_train,validation_split=0.2, epochs=10, batch_size=32)

        return history

    

    def VADER(self,unedited_dataframe,label_dataframe):
        
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
        sia=SIA()

        def sentiment_analyzer_scores(sentence):
            score = sia.polarity_scores(sentence)
            #print("{} {}".format(sentence, score))
            return score
        unedited_dataframe['VADER']=unedited_dataframe['Tweet'].apply(lambda x: sentiment_analyzer_scores(sentence=x))
        return unedited_dataframe


#%%

