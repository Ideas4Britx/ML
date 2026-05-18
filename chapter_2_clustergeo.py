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

class ClusterSimilarity(BaseEstimator,TransformerMixin):

    def __init__(self,n_clusters=10,gamma=1.0,random_state=None):
        self.n_clusters = n_clusters
        self.gamma=gamma
        self.random_state=random_state

    def fit(self,X,y=None,sample_weight=None):
        self.kmeans_=KMeans(self.n_clusters,n_init=10,random_state=self.random_state)
        self.kmenas_fit(X,sample_weight=sample_weight)

        return self

    def transform(self,X):
        return rbf_kernel(X,self.kmeans_.cluster_centers__,gamma=self.gamma)
    
    def get_feature_names_out(self,names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]