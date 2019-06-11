# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:53:58 2019

@author: Filip.Fraczek
"""

import numpy as np
import pandas as pd
import seaborn as sns


dane = pd.read_csv('train.csv')


brakujace = dane.loc[:, dane.isnull().mean() >= .2]
print(brakujace.columns)
train_data = dane.drop(columns = brakujace.columns)


dane['Name_length'] = dane['Name'].apply(len)
#test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic
dane['Has_Cabin'] = dane["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

type(dane.Cabin)
dane.loc[0:2,dane['Cabin'].dtype]

ilosc = pd.cut(dane['Fare'],4).uni
ilosc2 = pd.cut(dane['Fare'],4).unique()


dane['Title'] = dane.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dane['Title'] = dane['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
dane['Title'] = dane['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
dane['Title'] = dane['Title'].replace(['Lady'],'Mrs')

dane['Title'] = dane['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2, "Miss" : 3, "Mrs" : 4 })

dane = dane.drop('P_Ti_Age', axis=1)
###

dane['Has_age'] = dane['Age'].isnull().map(lambda x: 0 if x == True else 1)

wykres = sns.countplot(x='Pclass',hue='Has_age', data= dane)
# najwięcej ludzi ma wiek w klasie 3, najmniej w drugiej

wykres2 = sns.factorplot(y='Age',x= 'Pclass', data= dane, kind = 'point')
# w klasie 3 srednia wieku bardzo niska

dane['Title'].unique()

dane['Age2'] = dane['Age']

dane_pivottable2 = dane.pivot_table(values = 'Age', index = ['Pclass'],columns = ['Title'], aggfunc = np.median).values


for i in range(0,5):
    for j in range(1,4):
        dane.loc[(dane.Age.isnull()) & (dane.Pclass == j) & (dane.Title == i),'Age2'] = dane_pivottable2[j-1, i]

Ti_pred = dane.groupby('Title')['Age'].median().values


dane['Age3'] = dane['Age']

for i in range(0,5):
    dane.loc[(dane.Age.isnull()) & (dane.Title == i),'Age3'] = Ti_pred[i]





dane.Age.values

dane.Name.str.extract(' ([A-Za-z]+)\.')

### Jakie kolumny

for i in range (0,len(dane.columns)):
    wynik = str(i) + ':' + dane.columns[i]
    print(wynik)
    

dane.columns.index



## Sprawdzam poprawnosc danych ## podstawowe statystyki
dane.describe()

### 
print('Ilosc brakujacych danych w %')
tabela = []
col = []
zero = []
for i in range (0, len(dane.columns)):
    brak = np.array(dane.isnull().sum())
    brakuje = dane.columns[i] + ': ' + str(round(brak[i]/len(dane),2))+'%'
    
    kolumny = dane.columns[i]
    col.append(kolumny)
    
    braki = round(brak[i]/len(dane),2)
    zero.append(braki)
    
    print(brakuje)
    
    
zero = np.array(zero)    
nowy = list(zip(col,zero))
nowy2 = pd.DataFrame(nowy,columns=['zmienna','Nan'])
do_usuniecia = nowy2.loc[nowy2['Nan'] >= 0.3,]
train_data = dane.drop(columns = do_usuniecia.zmienna)


train_data








    
    

