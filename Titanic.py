#!# - tak oznaczam rzeczy, ktore zrobie pozniej (MP)
#!!# - punkt, w ktorym skonczylem

##### Import of the libraries #####

import numpy as np,  pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score 


##### Setting working directory #####

import os

##os.chdir('C:\\Users\Marek\\Desktop\\Python\\Kaggle\\Titanic')
os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Dane\\titanic')

##### Ignore warnings ##### 

import warnings
warnings.filterwarnings('ignore')

##### Related kernels #####

import webbrowser
webbrowser.open_new('https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy')

##### Import & Concat of the data ##### 

train = pd.read_csv('train.csv') #!# musi byc nazwa 'train', 'dane' mamy do zbioru 'train' + 'test'
test = pd.read_csv('test.csv')

dane = pd.concat((train, test)).reset_index(drop=True)

##### Available columns in the dataset #####

print("Kolumny jakie sa to: \n")
for i in range (0,len(dane.columns)):
    wynik = str(i) + ':' + dane.columns[i] #!# Survived ma indeks 10
    print(wynik, "\n")

##### Check of the basic descriptive statistics #####

dane.describe()
dane.head()
dane.tail()

##### Check of the variable types #####

print(dane.info()) 
dane.describe(include = 'all') 

##### Check how much data is missing #####

print("Ile % danych nam brakuje\n")
for i in range (0, len(dane.columns)):
    brak = np.array(dane.isnull().sum())
    missing = dane.columns[i] + ': ' + str(round(brak[i] * 100/len(dane),3))+'%'
    
    print(missing)
    
#!# -> wykres slupkowy z procentem missing Data

##### Grouping Survived by Sex #####

dane[['Sex','Survived']].groupby(['Sex'], as_index = True).mean().sort_values( by= 'Survived', ascending = False)

##### Grouping Survived by PClass #####

dane[['Pclass','Survived']].groupby(['Pclass'], as_index = True).mean().sort_values( by= 'Survived', ascending = False)

##### Dependence Survived ~ Sex & PClass (Factorplot) #####

g = sns.factorplot("Sex","Survived", hue = "Pclass", data=dane, kind="bar", palette="muted", legend=True)

##### Dependence Survived ~ Age (FacetGrid) #####

g = sns.FacetGrid(dane, col='Survived')
g.map(plt.hist, 'Age', bins=20)

##### Dependence Survived ~ PClass & Age (on separate FacetGrids) #####

g = sns.FacetGrid(dane, col = 'Survived' , row = 'Pclass')
g.map(plt.hist, 'Age')
g.add_legend()


##### Extract Title from Name & Check + replace of rare titles #####

dane['Title'] = dane.Name.str.extract('([a-zA-Z]+)\.')
dane.Title.unique()

for name in dane:
    dane['Title'] = dane['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dane['Title'] = dane['Title'].replace(['Mlle', 'Ms','Mme'], 'Miss')
    dane['Title'] = dane['Title'].replace('Mme', 'Mrs')

dane.Title.unique()

##### Convert titles to int datatype #####

dane.loc[dane.Title == "Mr", 'Title'] = int(0)
dane.loc[dane.Title == "Mrs", 'Title'] = int(4)
dane.loc[dane.Title == "Miss", 'Title'] = int(3)
dane.loc[dane.Title == "Rare", 'Title'] = int(1)
dane.loc[dane.Title == "Master", 'Title'] = int(2)

##### Convert Sex categories into int datatype #####

dane['Sex'] = dane['Sex'].apply(lambda x: int(1) if x == "male" else 0)

##### New column with info if a person has age (1) or not (0) #####

dane['Has_age'] = dane['Age'].apply(lambda x: 0 if str(x) == 'nan' else 1)

#### Chart -> people with age available ~ PClass #####

sns.set(style="darkgrid")
ax = sns.countplot(x="Pclass", hue = 'Has_age', data=dane)
ax.set_title("Ile osob ma podany wiek w zaleznosci od klasy")

#### Variables' correlation chart #####

korelacja = dane.corr(method='pearson')
corr = dane.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

##### Top 5 highest correlations #####

corr = corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool))
corr_sort = corr.unstack()
corr_sort = corr_sort.dropna(axis = 0)
corr_sort = corr_sort.sort_values(kind = 'quicksort')
abs(corr_sort).sort_values(kind='quicksort', ascending = False)

##### Pivot table with PClass & Title #####

Pclass_title_pred = pd.pivot_table(data=dane,values='Age', index = ['Pclass'],columns = ['Title'],aggfunc = np.median).values

dane['Has_age'] = dane['Age']

##### Filling in NA values in Has_Age with respective medians #####

for i in range(0,5):
    for j in range(1,4):
        dane.loc[(dane.Age.isnull()) & (dane.Title == i) & (dane.Pclass == j),'Has_age'] = Pclass_title_pred[j-1,i]

##### New Feature: Family Size #####

for data in dane:
    dane['Family_size'] = dane['SibSp'] + dane['Parch'] +1
    
##### Dependence between Family Size and probability of survival #####

family_survived = dane[['Family_size', 'Survived']].groupby(['Family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)

##### Survived ~ Embarked #####

kto_przezyl = dane.groupby(['Survived','Embarked']).size()
ax = sns.countplot( x = 'Embarked' , hue = 'Survived', data = dane)

dane.loc[dane['Embarked'].isnull() , ].index

##### Delete rows with Embarked = NA #####

dane = dane.dropna(axis=0, subset=['Embarked'])

##### Mapping integers to Embarked variable ######

dane['Embarked2'] = dane['Embarked'].map({ 'C' : 1, 'Q' : 2, 'S' : 3}).astype(int)

##### Clear dataset by dropping NULL Fare #####

dane['Fare'].isnull().sum()
dane['Fare'].dropna().median()
dane = dane.dropna(axis=0, subset=['Fare'])

##### Splitting Fare into 5 subsets #####

sns.distplot(dane['Fare']) #!# drobna sugestia - nazwa jest distplot, a tu mamy rozklad gestosci, dystrybuanta by rosla az do y = 1

dane['FareBand'] = pd.qcut(dane['Fare'], 4)
dane.loc[dane['Fare'] <= 7.896 , 'Fare' ] = 0
dane.loc[(dane['Fare'] > 7.896) & (dane['Fare'] <= 14.454), 'Fare' ] = 1
dane.loc[(dane['Fare'] > 14.454) & (dane['Fare'] <= 31.0), 'Fare' ] = 2
dane.loc[(dane['Fare'] > 31.0) & (dane['Fare'] <= 512.329), 'Fare' ] = 3
dane.loc[(dane['Fare'] > 512.329), 'Fare'] = 4

dane['Fare'] = dane['Fare'].astype(int)
dane['Fare'].isnull().sum()

#####  Dropping unnecessary columns ##### 

drop_columns = ['Age', 'Cabin', 'Embarked', 'Name', 'Ticket', 'FareBand','PassengerId']
dane = dane.drop(drop_columns, axis = 1)

X = dane.loc[:, dane.columns != 'Survived'].values
y = dane.iloc[:,0].values.astype(int)

##### Preparation for modelling #####

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X ,y,test_size = 0.2, random_state= 10)

##### Building Models #####

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

# Single models

# KNN 
knn = KNeighborsClassifier(n_neighbors=3, n_jobs = -1) #!# do optymalizacji parametry: n_neighbors (2,7), weights (uniform, distance), algorithm (auto, ball_tree, kd_tree, brute)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

# Confusion Matrix for KNN

knn_cm = confusion_matrix(y_test,Y_pred)
performance = int(round((knn_cm[0][0] + knn_cm[1][1]) / len(y_test),2) *100)
print('Performance KNN wynosi :', performance,'%')

# Random Forest
random = RandomForestClassifier(n_estimators = 20, max_depth = 10, random_state = 0, n_jobs = -1) #!# do optymalizacji: n_estimators, criterion, max_depth (?)  
classifier = random.fit(X_train, y_train)
predict = random.predict(X_test)

#Confusion Matrix for Random Forest 
rf_cm = confusion_matrix(y_test,predict)
performance = int(round((rf_cm[0][0] + rf_cm[1][1]) / len(y_test),2) *100) #!# unikajmy nazywania obiektow jak funkcje/metody (confusion_matrix)
print('Performance RF wynosi :', performance,'%')

# Grid Search for Random Forest

n_estimators = [int(x) for x in np.linspace(1,1000,10, endpoint = False)]
max_depth = [int(x) for x in np.linspace(1,100,10, endpoint = False)]

parameters = {'n_estimators' : n_estimators,
              'criterion' : ['gini','entropy'],
              'max_depth' : max_depth}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search_fit = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



##### BRUDNOPIS ##### 
    
    
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
    
#X_train[np.isnan(X_train).any(axis=1)].shape

