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


### DATA WRANGLING ###########################################################

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

### EDA: Exploratory Data Anaylisis ###########################################
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#General 
df.describe()
df.describe(include=['object']) #Sum of categorical variables
df.corr() #Correlation between all pair of numerical variables

#Between 2 quantitative variables (Dependent: price, independent: engine-size)
df[["engine-size", "price"]].corr() #Coeficiente correlación

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price']) #Coeficiente y p value por el método de Pearson
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

sns.regplot(x="engine-size", y="price", data=df) #Visualization of lineal regresion

#Categorical variables
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame() #To count values in 1 variable

sns.boxplot(x="body-style", y="price", data=df) #Visualization of distribution price along categories

#Categorical, GROUPING
group_1 = df[['price', 'drive-wheels','body-style']].groupby(['drive-wheels','body-style'],as_index=False).mean() #Groupby 2 variables
group_1_pivot = group_1.pivot(index='drive-wheels', columns='body-style') #Para ordenar a tabla más comprensible
group_1_pivot
group_1_pivot = group_1_pivot.fillna(0) #fill missing values with 0

plt.pcolor(group_1_pivot, cmap='Blues')
plt.colorbar() #Imprime la barra/leyenda
plt.show() #Visualization HEATMAP medias 'price' por categorías. Faltan los labels (complejo)

#Categ, grouping, ANOVA
group_2 = df[['price', 'drive-wheels']].groupby(['drive-wheels']) #No genera df????
group_2.head()
group_2.get_group('4wd')['price'] #Obtaining specific subgroups with method get_group
f_val, p_val = stats.f_oneway(group_2.get_group('fwd')['price'], group_2.get_group('rwd')['price'], group_2.get_group('4wd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)

### MODEL DEVELOPMENT #########################################################

### SLR (simple linear regression)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat = lm.predict(X)

new_input = np.arange(1,101,1).reshape(-1,1)
yhat_2=lm.predict(new_input) #To predict new values (for a range of X, i.e)

# MLR (multiple linear regression)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
lm.intercept_ #Value of intercept (Y at X=0)
lm.coef_#coefficients of each variable

### Model evaluation through visualization
sns.regplot(x='highway-mpg', y='price', data=df) #Regression plot
sns.residplot(df['highway-mpg'], df['price']) #Residual plot, showing different distribution of variance depending on highway-mpg

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted values", ax=ax1) #Distribution plot (actual vs. predicted values)

### Polynomial regression

#Using np.polyfit (for 1 dimension model)
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3) #Definition of a polynomial of the 3rd order (cubic) 
p = np.poly1d(f) #To find the coefficienst in polynomial
print(p)
PlotPolly(p, x, y, 'highway-mpg')

#Multivariate Polynomial regression (more than 1 variable). Not possible with np.polyfit
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2) 
Z_pr=pr.fit_transform(Z) #polynomial transform on multiple features
Z.shape
Z_pr.shape #From 4 features (Z) to 15 features

#Multivariate Polynomial. PIPELINE (to simplify transformation)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler #For normalization of variables
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())] #create the pipeline, by creating a list of tuples
pipe=Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]


### Model evaluation using in-sample measures

#SLR
lm.fit(X,Y)
print("R2 is ", lm.score(X,Y)) # R^2 value after fitting a linear regression model
print('The output of the first four predicted value is: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat) # MSE 

#MLR
lm.fit(Z, df['price'])
print('The R-square is: ', lm.score(Z, df['price'])) #Find the R^2
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], lm.predict(Z)))

#Polynomial fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(df['highway-mpg'])) #R2
mean_squared_error(df['price'], p(x)) #MSE
