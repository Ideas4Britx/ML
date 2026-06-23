from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import re

class GetTextFeaturesTransformer(BaseEstimator,TransformerMixin):

    puntuaction = [",", ".", ";", ":", "!", "?", "'", '"', 
              "-", "–", "—", "(", ")", "[", "]", "{", "}",
              "¿", "¡", "«", "»", "/", "@", "#", "$", 
              "%", "&", "*", "_", "^", "~", "|", "+", "=",
              "!","+"]

    def fit(self,X,y=None): #cuando es necesario aprender algo del dataset_train
        return self

    def transform(self,X):
        text_words =[]

        for text in X['text']:
            words_data = {}
            for word in text.split():
                isurl=re.search(r'https?://\S+|www\.\S+',word)#regex
                word = word.strip(''.join(self.puntuaction)).lower().strip() #strip al final quita todos los espacios
                if isurl:
                    if 'URL' not in words_data:
                        words_data['URL']=0
                    words_data['URL'] +=1
                elif word.isnumeric():
                    n_nums=len(word)
                    if 'NUM' not in words_data:
                        words_data['NUM']=0
                    words_data['NUM']+=n_nums
                else:
                    n_words = len(word)
                    if word not in words_data:
                        words_data[word] = 0#inicializo aqui, despues seimpre sumo
                    words_data[word] += 1
                    if 'CHAR' not in words_data:
                        words_data['CHAR']=0
                    words_data['CHAR']+=n_words

            text_words.append(words_data)

        X= pd.DataFrame(text_words).fillna(0)

        nums = X['NUM']

        chars = X['CHAR']

        X= X.drop(['NUM','CHAR'],axis=1)

        X_sum = X.sum(axis=1).rename('total_words')

        X = pd.concat([X,nums,chars,X_sum],axis=1)

        return X.to_numpy() #es muy sparse, muchos 0s. hay odelsoq eu se benfician de eso y entonces puede estar bien no escalar esas clases. 