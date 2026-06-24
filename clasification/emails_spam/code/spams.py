from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedShuffleSplit,RandomizedSearchCV,cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from custom_transform_words import GetTextFeaturesTransformer
from sklearn.metrics import classification_report,confusion_matrix
from scipy.stats import randint
import matplotlib.pyplot as plt

def spam_dataset() -> pd.DataFrame:
    principal_dir = Path.cwd()
    datasets_email = principal_dir.parent / 'dataset' / 'emails.csv'

    return pd.read_csv(datasets_email)

emails = spam_dataset().copy() #no hya ningun nulo

df = emails.groupby('spam').count()

print(df)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx, test_idx in split.split(emails,emails['spam']): #mismas clases train tesst
    strat_train_set=emails.loc[train_idx]
    strat_test_set = emails.loc[test_idx]

X_train = strat_train_set.drop('spam',axis=1) #train
y_train = strat_train_set['spam']#y

X_test = strat_test_set.drop('spam',axis=1)#predice sobre estos datos que no ve en entrenamiento
y_test = strat_test_set['spam']#comparar para si produce bien

#Xtrain e Xterst titnene que tener las mismas columnas y tamanyo, igual que y.

#mirar cuanto es spam y cuanto no es spam en el dataset original



puntuaction = [",", ".", ";", ":", "!", "?", "'", '"', 
            "-", "–", "—", "(", ")", "[", "]", "{", "}",
            "¿", "¡", "«", "»", "/", "@", "#", "$", 
            "%", "&", "*", "_", "^", "~", "|", "+", "=",
            "!","+"]


default_num_pipeline = make_pipeline(FunctionTransformer(np.log1p),StandardScaler())

preprocessing = ColumnTransformer([
    ('get_count_words',GetTextFeaturesTransformer(),['text']),
    ('normalize',default_num_pipeline,make_column_selector(dtype_exclude=object))
],remainder="drop")

pipeline = make_pipeline(preprocessing,RandomForestClassifier(random_state=42))

#scores = cross_val_score(pipeline,X_train,y_train,cv=5,scoring='f1')

#print(scores.mean())

#print(scores.std()) #consistencia entre foldds

#y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=5)

#print(classification_report(y_train, y_train_pred))

param_grid = {
    'randomforestclassifier__n_estimators': randint(100, 500),
    'randomforestclassifier__max_depth': randint(3, 20),
    'randomforestclassifier__min_samples_split': randint(2, 20),
    'randomforestclassifier__min_samples_leaf': randint(1, 10),
    'randomforestclassifier__max_features': ['sqrt', 'log2']
}

rnd_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

rnd_search.fit(X_train, y_train)

best_pipeline = rnd_search.best_estimator_

y_test_pred = best_pipeline.predict(X_test)

print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))