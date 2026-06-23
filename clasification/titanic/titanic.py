from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


def load_titanic_data():

    tarball_path = Path('datasets/titanic.tgz')

    if not tarball_path.is_file():#descargarmos
        Path('datasets').mkdir(parents=True,exist_ok=True)
        url = 'https://homl.info/titanic.tgz'
        urllib.request.urlretrieve(url,tarball_path)
    
    with tarfile.open(tarball_path) as titanic_tarball:
        titanic_tarball.extractall(path='datasets')
        
        test_csv = Path('datasets/titanic/test.csv')
        train_csv = Path('datasets/titanic/train.csv')
  

        return pd.read_csv(train_csv),pd.read_csv(test_csv)

   

train,test = load_titanic_data() # viene listo con el target en train dropped

def drop_columns(X, columns):
    X = X.copy()
  
    return X.drop(columns, axis=1)

def total_family(X):
    X = X.copy()
    
    X['total_family'] = X['Parch'] + X['SibSp']

    return X

def has_cabin(X):
    X = X.copy()
    X['Has_Cabin'] = X['Cabin'].notnull().astype(int)
    return X

# Train
X = total_family(train)
X = has_cabin(X)
X = drop_columns(X, ['Cabin','Parch','SibSp','Pclass'])

# Test
test = test.copy()
test = total_family(test)
test = has_cabin(test)
test = drop_columns(test, ['Cabin','Parch','SibSp','Pclass'])

# Target
y = X['Survived']
X = X.drop('Survived', axis=1)

print(test.info())

#a partir de ahora train es nuestro X
    
def no_skwed_fare_pipeline():
    return make_pipeline(FunctionTransformer(np.log1p),StandardScaler())


default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

default_str_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore',sparse_output=False))


preprocessing = ColumnTransformer([
    ('price_ticket',no_skwed_fare_pipeline(),['Fare']),
    ('num', default_num_pipeline, ['Age', 'total_family', 'Has_Cabin']),
    ('cat', default_str_pipeline, make_column_selector(dtype_include=object))
],
remainder='drop') 

pipeline = make_pipeline(
    preprocessing,
    RandomForestClassifier()
)


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
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

rnd_search.fit(X, y)
print('best params',rnd_search.best_params_)
print('best socre',rnd_search.best_score_)

best_pipeline = rnd_search.best_estimator_

# Con el mejor modelo
y_probas = cross_val_predict(best_pipeline, X, y, cv=5, method='predict_proba')[:,1]
precisions, recalls, thresholds = precision_recall_curve(y, y_probas)

# y_pred_train = cross_val_predict(best_pipeline, X, y, cv=5)

# cm = confusion_matrix(y, y_pred_train)
# print(cm)

#esto d aqui usa el threshold mediano entonces siempre se basa en 0.5

# Buscar threshold para 90% recall
idx = (recalls < 0.90).argmax() - 1
threshold = thresholds[idx]
print(f"Threshold: {threshold}")
print(f"Precision: {precisions[idx]:.2f}")
print(f"Recall: {recalls[idx]:.2f}")

y_pred_custom = (y_probas >= threshold).astype(int) 

cm = confusion_matrix(y, y_pred_custom)#custom thershold
print(cm)

# Predecir test con ese threshold
y_test_probas = best_pipeline.predict_proba(test)[:,1]
y_pred = (y_test_probas >= threshold).astype(int)

print(y_pred)
print(f"Supervivientes predichos: {y_pred.sum()}")
print(f"Muertos predichos: {(y_pred==0).sum()}")


# plt.plot(thresholds,precisions[:-1],"b--",label='Precision',linewidth=2)
# plt.plot(thresholds,recalls[:-1],'g-',label='Recall',linewidth=2)
# plt.vlines(threshold_for_90_recall, 0, 1.0, "k", 'dotted', label='threshold')
# plt.show()

#tener laa oportuniad de probar roi