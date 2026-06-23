from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor #tambien en clasifier
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer,make_column_selector,make_column_transformer
from chapter_2_clustergeo import ClusterSimilarity
from scipy.stats import randint

def load_housing_data(): #al ser un proceso recurrente si se modifrica o actualiza el modelo,crear funcion que descargue la data y la cargue!
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

def which_skewed():#fn para saber quienes estan skewed
    pass

def matrix_correlation():
    corr_matrix = housing.drop("ocean_proximity",axis=1).corr()
    return str(corr_matrix['median_house_value'].drop('median_house_value').idxmax()) #columna maximo quitando el target

def group_data_by_correlation(housing, strat_field, best_corr_feature): #lo mejor es copiar el df
    df = housing.copy()  # no tocar el original
    df[f'{strat_field}_strat'] = pd.cut(df[best_corr_feature],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5])
    return df

housing = group_data_by_correlation(housing, "income", matrix_correlation())
print(housing.info())

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx,test_idx in split.split(housing,housing['income_strat']):
    strat_train_set = housing.loc[train_idx]# 80% de los datos, da los datos de cafa fila y los va cogiendo en un dataset
    strat_test_set = housing.loc[test_idx]

strat_train_set = strat_train_set.drop("income_strat", axis=1) #dropeamos campos para hacer 
strat_test_set = strat_test_set.drop("income_strat", axis=1)

# Extraer labels y features
housing_labels = strat_train_set["median_house_value"].copy()
housing_train = strat_train_set.drop("median_house_value", axis=1)

housing_test_labels = strat_test_set["median_house_value"].copy()
housing_test = strat_test_set.drop("median_house_value", axis=1)

def column_ratio(X):
    return X[:,[0]] / X[:,[1]]


def ratio_name(function_transformer,feature_name_in):
     return ["ratio"] #features_name_out


def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy='median'),
    FunctionTransformer(column_ratio,feature_names_out=ratio_name),
    StandardScaler()) #normalizar la escala


log_pipeline = make_pipeline(SimpleImputer(strategy='median'),
FunctionTransformer(np.log,feature_names_out="one-to-one"),StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10,gamma=1,random_state=42)

default_num_pipeline= make_pipeline(SimpleImputer(strategy='median'),StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

#housing_prepared = preprocessing.fit_transform(strat_train_set)
#housing_test_prepared = preprocessing.transform(strat_test_set)

#forest_reg = make_pipeline(preprocessing,RandomForestRegressor(random_state=42))
#forest_rmses = -cross_val_score(forest_reg,housing_train,housing_labels,scoring='neg_root_mean_squared_error',cv=10)

#forest_reg.fit(housing_train, housing_labels)

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("randomforestregressor", RandomForestRegressor(random_state=42))
])



param_distribs = {'preprocessing__geo__n_clusters':randint(low=3,high=50),
                'randomforestregressor__max_features':randint(low=2,high=20)}

rnd_search = RandomizedSearchCV(full_pipeline,param_distributions=param_distribs,n_iter=10,cv=3,
            scoring='neg_root_mean_squared_error',random_state=42)

rnd_search.fit(housing_train,housing_labels)#tambien hace transform

best_model = rnd_search.best_estimator_ #mejor modelo encontrado con los hiperparametros

best_cv_rmse = int(-rnd_search.best_score_)
print(f"Best CV RMSE: {best_cv_rmse}")

housing_test_predictions = best_model.predict(housing_test)
rmse_test = root_mean_squared_error(housing_test_labels, housing_test_predictions)
print(f"RMSE test: {rmse_test:.0f}")