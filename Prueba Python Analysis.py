# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:12:37 2019

@author: helen
"""

import pandas as pd
import os
print(os.getcwd())

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None) #Este csv no tiene headers (a especificar)
print(df.head(5)) #para enseñar las 5 primeras líneas

headers = ["symboling","normalized-loss","make","fuel-type","aspiration","num-doors","body-style", "drive-wheels", "engine-location","wheel-base", "lenght","width","height","curb-weight", "engine-type","num-cyl", "engine-size","fuel-ssytem", "bore", "stroke", "comp-ratio", "horse", "peak-rpm", "city-mpg","highway-mpg","price"]
df.columns = headers
print(df.head(5)) #Ahora ya tienen nombre las columnas

df.to_csv('PythonTest.csv')

#Basic insights of Dataset
df.dtypes #Para comproar el tipo de las variables
df.describe() #Statistical summary of numeric-typed columns
df.describe(include="all") #Full summary, including type object
df[["width","engine-size"]].describe() #To select columns (atención! 2 corchetes!)
df.info() #Provides all info showing first and last 30 rows del df


## DATA WRANGLING

#Missing values
df.dropna(subset=["price"], axis=0, inplace=True) #Drop the entire row with mossing price. No funciona porque price no es numérico
mean = df["curb-weight"].mean()
df["curb-weight"].replace(np.nan, mean) #Replace by the mean.np not defined

#Data formatting
df["city-mpg"] = 235/df["city-mpg"] #Convertir una variable a otra unidad, sustituyéndola
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True) #Y renombrándola
print(df["city-mpg"].head(5)) #Error porque ya no existe

df["price"] = df["price"].astype("int") # Cambio de type. No funciona
