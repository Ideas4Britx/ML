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


#first we look to the data, a preliminary view! 

def load_housing_data(): #al ser un proceso recurrente si se modifrica o actualiza el modelo,crear funcion que descargue la data y la cargue!
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data() #return csv dataset y lo carga en andas21

print(housing.head()) #headers con algunas lineas dde ejemplo

print(housing.info()) #nos otorgua cada nombre de columna, los tipos, si hay nulos, y el conteo total

print(housing['ocean_proximity'].value_counts()) #cuenta cada tipo de valor.

print(housing.describe()) #describe el sumatorio de los atributos numericos

#housing.hist(bins=50, figsize=(12,8))
#plt.show() #histograma, distribuciones skewed righ.m atributos con diferntes escalas. 

#secondly we just separate the test set form the data, without manipulatin anythin.


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5]) #para separar mejor en train los datos
"""

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n]) #los splits se crean para asi poder evaluar la estbailidad de la mediad e precision. 

    creo que es mejor hacer splits haciendolo de manera equitataiba con los datos
"""




strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)


housing = strat_train_set.copy()

#housing.plot(kind="scatter", x="longitude", y='latitude', grid = True, alpha = 0.2)#do0nde resinden la mayoria de los puntos, mas densidad

#housing.plot(kind="scatter",x="longitude", y='latitude', grid = True,s=housing['population']/100, label = 'population',c="median_house_value",cmap="jet",colorbar=True, legend=True, sharex= False,
#figsize=(10,7)) #nbos fijamos en los puntos con mas densidad el valor de precio y tambien la grandaria del punto, apra asi ver la population y precio. s es tama;o puntos 

"""

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True) #enfatiamos en donde hay mas correlacion

plt.show()

"""

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"] #combinacion de atributos con mas interes que los mismos sueltos

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False) 


#print(housing.info)

#cleaning the data

#cleaning NA from total bedrooms

null_rows_idx = housing.isnull().any(axis=1)#buscas filas con nulkos
housing.loc[null_rows_idx].head()# miras filas una vez

housing_option3 = housing.copy()

median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3

median = housing["bedrooms_ratio"].median()
housing_option3["bedrooms_ratio"].fillna(median, inplace=True)#hubiera sido mejor antes de hacer este claculo dejharlo todo no nulo? el total bedreooms? 

housing_option3.loc[null_rows_idx].head() #miras las filas que habian atneriormente nulos, pero no deberias ver ninguno.

housing_cat = housing_option3[["ocean_proximity"]] #aqui lo que hacemos es 
housing_cat.head(8)

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)#en este caso, la distancia entre numero x a numero y hace que el ml crea que tiene cierta correlacion

cat_encoder = OneHotEncoder()#mejor usar este
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

df_output = pd.DataFrame(
    housing_cat_1hot.toarray(),
    columns=cat_encoder.get_feature_names_out(),
    index=housing_cat.index
)


housing_final = housing_option3.drop("ocean_proximity", axis=1).copy()

housing_final = pd.concat([housing_final, df_output], axis=1) #concatener campos hot encoded con dataset entero. habrai que remover la variable str creo

print(housing_final.info()) #podria dropear la carteagoria dedl income, que solo sirve realmente para hacer el test y demas. 
 
#arreglar todo este merder. 
#fit sobre el train y despues transform en test y train???

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

housing_final['population'].hist(ax=axes[0],bins=50)#ivide los valores en rangos (bins) y cuenta cuántos datos caen en cada rango. El eje X son los valores, el eje Y es la frecuencia.
axes[0].set_title("Population - Original")

# Transformada
np.log(housing["population"]).hist(ax=axes[1], bins=50)
axes[1].set_title("Population - Log")

skew_dataset= housing_final['population'].skew()#obviamente solo para datos numericos

print(skew_dataset)#3.5 muy pronunciao a la derecha. el singo da para donde tira.  

kde = gaussian_kde(housing["housing_median_age"])
x = np.linspace(housing["housing_median_age"].min(), housing["housing_median_age"].max(), 1000)
density = kde(x)

# Índices donde hay picos
peak_indices = np.where((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]))[0]

# Valor de la feature en cada pico
peak_values = x[peak_indices]
peak_densities = density[peak_indices]

for val, den in zip(peak_values, peak_densities):
    print(f"Pico en: {val:.1f} con densidad: {den:.4f}")

