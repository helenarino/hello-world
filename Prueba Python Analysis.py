# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:12:37 2019

@author: helen
"""

import pandas as pd
import numpy as np

#Find the path
import os
print(os.getcwd())

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#Importar archivo y generar dataframe
df = pd.read_csv(url, header = None) #Este csv no tiene headers (a especificar)
print(df.head(5)) #para enseñar las 5 primeras líneas

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
print(df.head(5)) #Ahora ya tienen nombre las columnas

#Basic insights of Dataset
df.dtypes #Para comproar el tipo de las variables
df.describe() #Statistical summary of numeric-typed columns
df.describe(include="all") #Full summary, including type object
df[["width","engine-size"]].describe() #To select columns (atención! 2 corchetes!)
df.info() #Provides all info showing first and last 30 rows del df


## DATA WRANGLING

#Missing values
#Finding
df.replace("?", np.nan, inplace=True) #Para convertir los missing en formato manejable por python
missing_data = df.isnull() #Generates sparse matrix F/T (true ==missing)
for column in missing_data.columns:
    print("\n", column)
    print(missing_data[column].value_counts()) #Te enseña cuántos missing por variable
    
#Dealing with missing:drop variables
df.dropna(subset=["price"], axis=0, inplace=True) #Drop the entire row with mossing price
df.reset_index(drop=True, inplace=True) #Reset index
#Substitute missing by mean ("normalized-losses", "stroke", "bore", "horsepower", "peak-rpm")
#avg_nl = df["normalized-losses"].mean() #No funciona porque type object. 
avg_nl = df["normalized-losses"].astype("float").mean()
df["normalized-losses"].replace(np.nan, avg_nl, inplace=True) #Replace by the mean
df["normalized-losses"] = df["normalized-losses"].astype("int") #Correción del tipo de variable a entero (estaba en object)

#Para el resto de variables, cambio primero el tipo a float, hago media y sustituyo
df[["stroke", "bore", "horsepower", "peak-rpm"]] = df[["stroke", "bore", "horsepower", "peak-rpm"]].astype("float")
avg_str = df["stroke"].mean()
avg_bore = df["bore"].mean()
avg_hpw = df["horsepower"].mean()
avg_prpm = df["peak-rpm"].mean()
df["stroke"].replace(np.nan, avg_str, inplace=True)
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["horsepower"].replace(np.nan, avg_hpw, inplace=True)
df["peak-rpm"].replace(np.nan, avg_prpm, inplace=True)

df.info()#Still num-of-doors has NaNs
df['num-of-doors'].value_counts()
df["num-of-doors"].replace(np.nan, df['num-of-doors'].value_counts().idxmax(), inplace=True) #Sustituirlo por el valor más frec "four"

#Data formatting
df["city-mpg"] = 235/df["city-mpg"] #Convertir una variable a otra unidad, sustituyéndola
df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True) #Y renombrándola
print(df["city-mpg"].head(5)) #Error porque ya no existe

df["price"] = df["price"].astype("int") #Era object

#Data formatting of indicators (or dummy variables: few categorical options)
df["fuel-type"].describe()
dummy_variable_1 = pd.get_dummies(df["fuel-type"]) #Genera una nueva matriz con 2 variables 1/0. Hay que merge con la original
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1) #Concat por columnas
df.drop("fuel-type", axis = 1, inplace=True) #Elimino la original, ahora redundante

dummy_variable_2 = pd.get_dummies(df["aspiration"]) #Genera una nueva matriz con 2 variables 1/0. Hay que merge con la original
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1) #Concat por columnas
df.drop("aspiration", axis = 1, inplace=True)

#Normalization
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

#Binning (ex categorization of numerical variables)
bins_hpw = np.linspace(min(df['horsepower']), max(df['horsepower']), 4) #Genera 3 bins (entre 4 cortes)
df['horsepower-binned'] = pd.cut(df['horsepower'], bins_hpw, labels=['Low', 'Medium', 'High'], include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

#Guardar
df.to_csv('PythonTest.csv') 
