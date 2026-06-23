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

    def fit(self, X, y=None):
        vocab = set()
        for text in X['text']:
            for word in text.split():
                isurl = re.search(r'https?://\S+|www\.\S+', word)
                word = word.strip(''.join(self.puntuaction)).lower().strip()
                if isurl:
                    continue
                elif word.isnumeric():
                    continue
                else:
                    vocab.add(word)

        self.word_vocabulary_ = sorted(vocab)
        self.extra_features_ = ['URL', 'NUM', 'CHAR', 'total_words']
        self.vocabulary_ = self.word_vocabulary_ + self.extra_features_
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

        X_df = pd.DataFrame(text_words).fillna(0)

        # total_words: lo calculas ANTES del reindex, sumando solo las columnas de palabras reales
        word_cols = [c for c in X_df.columns if c in self.word_vocabulary_]
        X_df['total_words'] = X_df[word_cols].sum(axis=1)

        # aquí se fuerza todo a las columnas fijas aprendidas en fit
        X_df = X_df.reindex(columns=self.vocabulary_, fill_value=0)

        return X_df.to_numpy() #es muy sparse, muchos 0s. hay odelsoq eu se benfician de eso y entonces puede estar bien no escalar esas clases. 