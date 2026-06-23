from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,FunctionTransformer
import matplotlib.pyplot as plt
from custom_transform_words import GetTextFeaturesTransformer

def spam_dataset() -> pd.DataFrame:
    principal_dir = Path.cwd()
    datasets_email = principal_dir.parent / 'dataset' / 'emails.csv'

    return pd.read_csv(datasets_email)

emails = spam_dataset().copy() #no hya ningun nulo

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx, test_idx in split.split(emails,emails['spam']): #mismas clases train tesst
    strat_train_set=emails.loc[train_idx]
    strat_test_set = emails.loc[test_idx]

X_train = strat_train_set.drop('spam',axis=1) #train
y_train = strat_train_set['text']#y

X_test = strat_test_set['text']#predice sobre estos datos que no ve en entrenamiento
y_test = strat_test_set['spam']#comparar para si produce bien


puntuaction = [",", ".", ";", ":", "!", "?", "'", '"', 
            "-", "–", "—", "(", ")", "[", "]", "{", "}",
            "¿", "¡", "«", "»", "/", "@", "#", "$", 
            "%", "&", "*", "_", "^", "~", "|", "+", "=",
            "!","+"]


default_num_pipeline = make_pipeline(FunctionTransformer(np.log1p),StandardScaler())

preprocessing = ColumnTransformer([
    ('get_count_words',GetTextFeaturesTransformer(),['text']),
    ('normalize',default_num_pipeline,make_column_selector(dtype_exclude=object))
])

pipline = make_pipeline(preprocessing,RandomForestClassifier(random_state=42))

