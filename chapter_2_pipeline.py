from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, MinMaxScaler
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

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

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx,test_idx in split.split(housing,housing['income_strat']):
    strat_train_set = housing.loc[train_idx]# 80% de los datos, da los datos de cafa fila y los va cogiendo en un dataset
    strat_test_set = housing.loc[test_idx]

strat_train_set = strat_train_set.drop("income_strat", axis=1) #dropeamos campos para hacer 
strat_test_set = strat_test_set.drop("income_start", axis=1)

def column_ratio(X):
    return X[:,[0]] / X[:,[1]]