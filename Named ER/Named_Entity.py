
# %%
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
class SpaCy:
    
    def NER(self,text,display=False):
        nlp=spacy.load("en_core_web_sm") 

        Ent=list()
        
        # loading the english small model 
        ruler=EntityRuler(nlp)
        # adding the entity ruler for custom entity tagging
        # use pattern=[{"label":"label_to assign as","pattern":"pattern_or entity to recognise"}, and so on]
        pattern=[{"label":"label",'pattern':'pattern'
        }]
        ruler.add_patterns(pattern)
        nlp.add_pipe(ruler)

        docs=nlp.pipe(text,disable=["tagger", "parser"])
        for doc in docs:
            k=[(ent.text,ent.label_) for ent in doc.ents]
            Ent.append(k)

        if display: # if you want to display the sentence with the named entities highlighted
            displacy.render(doc, jupyter=True, style='ent')
        return Ent #returns a list of lists that contain tuples for (entity,label)
        
    def ORG(self,tokenized_tuple_sentence,display=False): #to check if the entity is a organisation
        #we may not have need for this function if we do overwrite_ents=True in the EntityRuler where ruler is defined
        org=[]
        for _tuple in tokenized_tuple_sentence:
            text,ent=_tuple
            if ent=='ORG':
                org.append(text)
        #returns a list of organisations that are present and were recognised by spacy
        return org  

#***********************************************************************8
#%%
import pandas as pd 
spcy=SpaCy()
df=pd.read_csv(r'C:\Users\Samay Panwar\OneDrive - Nanyang Technological University\INTERN\naive bayes model\Apple_DataFrame.csv')
df=df.iloc[:120]

Ent=spcy.NER(df.Tweet)
df['Entities']=Ent
df['Organisation']=df['Entities'].apply(lambda x: spcy.ORG(x))
df.head()


#NER does not work for lower case words
#where to put and how to apply emoji analysis 
#problem with function in emoji sentiment
#remove emoji in the same function as sentiment

#upload to github
#google cloud function

