import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Ustalenie sciezki roboczej

import os
#os.chdir('C:\\Users\Marek\\Desktop\\Python\\Kaggle\\Titanic')
#Ignorowanie ostrzezen
import warnings
warnings.filterwarnings('ignore')

#Literatura pomocnicza
import webbrowser
webbrowser.open_new('https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy')

#Wczytanie danych
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

dane = pd.concat((train, test)).reset_index(drop=True)

### Jakie kolumny mam w zbiorze danych

print("Kolumny jakie sa to: \n")
for i in range (0,len(dane.columns)):
    wynik = str(i) + ':' + dane.columns[i]
    print(wynik, "\n")

## Sprawdzam podstawowe statystyki

dane.describe()
dane.head()
dane.tail()

## Typy zmiennych

print(dane.info()) 
dane.describe(include = 'all') 



## Pokaz ile brakuje danych
print("Ile % danych nam brakuje\n")
for i in range (0, len(dane.columns)):
    brak = np.array(dane.isnull().sum())
    missing = dane.columns[i] + ': ' + str(round(brak[i]/len(dane),3))+'%'
    
    print(missing)
    
## grupowanie przy pomocy płci
dane[['Sex','Survived']].groupby(['Sex'], as_index = True).mean().sort_values( by= 'Survived', ascending = False)
# grupowanie przy pomocy klasy
dane[['Pclass','Survived']].groupby(['Pclass'], as_index = True).mean().sort_values( by= 'Survived', ascending = False)

# Używam facetgrid aby narysować wykres z zależnościami 
# Set up a factorplot z biblioteki seaborn # rodzaje {point, bar, count, box, violin, strip}
g = sns.factorplot("Sex","Survived", hue = "Pclass", data=dane, kind="bar", palette="muted", legend=True)

#Wykres pokazujacy zaleznosc miedzy przezyciem a wiekiem
g = sns.FacetGrid(dane, col='Survived')
g.map(plt.hist, 'Age', bins=20)

#wykres pokazujacy przezycie w zaleznosci od klasy i wieku
g = sns.FacetGrid(dane, col = 'Survived' , row = 'Pclass')
g.map(plt.hist, 'Age')
g.add_legend()

# Wyciągam tytul z imienia
# niestety ale ta funkcja jest bezuzyteczna gdyz w indexie = 18 mamy osobe ktora ma 2 imiona
#string = dane.Name.str.split()
#tytul = list(string)

#zmienna = []
#for i in range(0, len(tytul)):
#    wynik = tytul[i][1]
#    zmienna.append(wynik)

#zmienna = [word.replace('.','') for word in zmienna]
#dane['Title'] = zmienna



####### uzyway funkcje z biblioteki re czyli regular expression

#Wyciagam tytul przy imieniu do nowej zmiennej 'Title'

dane['Title'] = dane.Name.str.extract('([a-zA-Z]+)\.')
# unikalne wartosci dla kolumny Title
dane.Title.unique()

for name in dane:
    dane['Title'] = dane['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dane['Title'] = dane['Title'].replace(['Mlle', 'Ms','Mme'], 'Miss')
    dane['Title'] = dane['Title'].replace('Mme', 'Mrs')

dane.Title.unique()

# zamieniam tekst w int
dane.loc[dane.Title == "Mr", 'Title'] = int(0)
dane.loc[dane.Title == "Mrs", 'Title'] = int(4)
dane.loc[dane.Title == "Miss", 'Title'] = int(3)
dane.loc[dane.Title == "Rare", 'Title'] = int(1)
dane.loc[dane.Title == "Master", 'Title'] = int(2)

#zamieniam male i female na int

dane['Sex'] = dane['Sex'].apply(lambda x: int(1) if x == "male" else 0)

# tworze kolumne czy ma wiek
dane['Has_age'] = dane['Age'].apply(lambda x: 0 if str(x) == 'nan' else 1)

# wykres pokazujacy ile osob ma wiek w zaleznosci od klasy

sns.set(style="darkgrid")
ax = sns.countplot(x="Pclass", hue = 'Has_age', data=dane)
ax.set_title("Ile osob ma podany wiek w zaleznosci od klasy")

# Korelacje
korelacja = dane.corr(method='pearson')

corr = dane.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)


# Tworze tabele przestawna

Pclass_title_pred = pd.pivot_table(data=dane,values='Age', index = ['Pclass'],columns = ['Title'],aggfunc = np.median).values

dane['Has_age'] = dane['Age']

#Uzupelniam zmienna 'Has_Age' mediana z kolumny przestawnej.
for i in range(0,5):
    for j in range(1,4):
        dane.loc[(dane.Age.isnull()) & (dane.Title == i) & (dane.Pclass == j),'Has_age'] = Pclass_title_pred[j-1,i]


# wielkosc rodziny

for data in dane:
    dane['Family_size'] = dane['SibSp'] + dane['Parch'] +1
    
# jak wielkosc rodziny wplywa na przezycie
family_survived = dane[['Family_size', 'Survived']].groupby(['Family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# chce zmienic cene biletu w zmienna numeryczna

#Wykres pokazujacy zaleznosc miedzy przezyciem a wiekiem
#Analiza atrybutu embarked
kto_przezyl = dane.groupby(['Survived','Embarked']).size()
ax = sns.countplot( x = 'Embarked' , hue = 'Survived', data = dane)

dane.loc[dane['Embarked'].isnull() , ].index

# usuwam 2 wiersze gdzie embarked = null
dane = dane.dropna(axis=0, subset=['Embarked'])

# wrzucam dane
dane['Embarked2'] = dane['Embarked'].map({ 'C' : 1, 'Q' : 2, 'S' : 3}).astype(int)

#Analiza fare
dane['Fare'].isnull().sum()
# zero nulli
dane['Fare'].dropna().median()

#Sprawdzam dystrybuante
sns.distplot(dane['Fare'])

#dziele Fare na 4 rowne podzbiory
dane['FareBand'] = pd.qcut(dane['Fare'], 4)

dane.loc[dane['Fare'] <= 7.896 , 'Fare' ] = 0
dane.loc[(dane['Fare'] > 7.896) & (dane['Fare'] <= 14.454), 'Fare' ] = 1
dane.loc[(dane['Fare'] > 14.454) & (dane['Fare'] <= 31.0), 'Fare' ] = 2
dane.loc[(dane['Fare'] > 31.0) & (dane['Fare'] <= 512.329), 'Fare' ] = 3
dane.loc[(dane['Fare'] > 512.329), 'Fare'] = 4
dane['Fare'] = dane['Fare'].astype(int)

dane['Fare'].isnull().sum()
dane = dane.dropna(axis=0, subset=['Fare'])

# Usuniecie zbednych kolumn
drop_columns = ['Age', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBand','PassengerId']
dane = dane.drop(drop_columns, axis = 1)


X = dane.loc[:, dane.columns != 'Survived'].values
y = dane.iloc[:,5].values.astype(int)


#Splitting Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X ,y,test_size = 1/3, random_state= 10)


np.unique(y_train[0])


X_train[np.isnan(X_train).any(axis=1)].shape

# building models

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

names = []
results = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train,y_train,cv=10,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)



    

