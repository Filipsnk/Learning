##### Import of the libraries #####

import numpy as np,  pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
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

import os

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
    print(wynik)

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
    
##### Storing the missing data in new variables ##### 

all_data_na = (dane.isnull().sum() / len(dane)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

##### Barplot with percentage of missing data #####

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

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

sns.distplot(dane['Fare']) 
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
y = dane.iloc[:,5].values.astype(int)
X_train_start = X[0:889,:]
y_train_start = y[0:889] ### zbior dane laczy zbiory train oraz test. Dla zbioru test cala kolumna Survived 

##### Preparation for modelling #####

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_train_start,y_train_start,test_size = 0.2, random_state= 10)

##### Building Models #####

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RTC', RandomForestClassifier()))

names = []
results = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train,y_train,cv=10,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)

##### Single models ######

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
Y_pred = lr.predict(X_test)
acc_lr = round(lr.score(X_train, y_train) * 100, 2)

# Confusion Matrix for LR

lr_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((lr_cm[0][0] + lr_cm[1][1]) / len(y_test),5) *100)
print('Performance LR wynosi :', performance,'%')

# Linear Discriminant Analysis 

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
Y_pred = lda.predict(X_test)
acc_lda = round(lda.score(X_train, y_train) * 100, 2)

# Confusion Matrix for LDA

lda_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((lda_cm[0][0] + lda_cm[1][1]) / len(y_test),5) *100)
print('Performance LDA wynosi :', performance,'%')

# K Nearest Neighbors
 
knn = KNeighborsClassifier(n_neighbors=3, n_jobs = -1)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

# Confusion Matrix for KNN

knn_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((knn_cm[0][0] + knn_cm[1][1]) / len(y_test),5) *100)
print('Performance KNN wynosi :', performance,'%')

# Decision Tree Classifier
 
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
Y_pred = dtc.predict(X_test)
acc_dtc = round(dtc.score(X_train, y_train) * 100, 2)

# Confusion Matrix for DTC

dtc_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((dtc_cm[0][0] + dtc_cm[1][1]) / len(y_test),5) *100)
print('Performance DTC wynosi :', performance,'%')

# Gaussian NB
 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
Y_pred = gnb.predict(X_test)
acc_gnb = round(gnb.score(X_train, y_train) * 100, 2)

# Confusion Matrix for GNB

gnb_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((gnb_cm[0][0] + gnb_cm[1][1]) / len(y_test),5) *100)
print('Performance GNB wynosi :', performance,'%')

# SVC
 
svc = SVC() #!# do optymalizacji parametry: n_neighbors (2,7), weights (uniform, distance), algorithm (auto, ball_tree, kd_tree, brute)
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)

# Confusion Matrix for SVC

svc_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((svc_cm[0][0] + svc_cm[1][1]) / len(y_test),5) *100)
print('Performance SVC wynosi :', performance,'%')

# Random Forest
rf = RandomForestClassifier() 
rf.fit(X_train, y_train)
predict = rf.predict(X_test)

#Confusion Matrix for Random Forest 
rf_cm = confusion_matrix(y_test,predict)
performance = float(round((rf_cm[0][0] + rf_cm[1][1]) / len(y_test),5) *100) 
print('Performance RF wynosi :', performance,'%')

##### Grid Search for the best model - K-Nearest Neighbors #####

### Parameters for tuning process ###

n_neighbors = list(range(1,11))
weights =  ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

parameters = {'n_neighbors' : n_neighbors,
              'weights' : weights,
              'algorithm' : algorithm,
              'metric' : metric}

grid_search = GridSearchCV(estimator = knn, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train) 

best_accuracy = grid_search.best_score_
print(best_accuracy)
best_parameters = grid_search.best_params_
print(best_parameters)

### KNN tuned with Grid Search ###

knn_gs = KNeighborsClassifier(algorithm = 'brute', metric = 'euclidean',
                           n_neighbors = 9, weights = 'uniform', n_jobs = -1)
knn_gs.fit(X_train, y_train)
Y_pred = knn_gs.predict(X_test)
acc_knn = round(knn_gs.score(X_train, y_train) * 100, 2)

# Confusion Matrix for KNN tuned with Grid Search

knn_gs_cm = confusion_matrix(y_test,Y_pred)
performance = float(round((knn_gs_cm[0][0] + knn_gs_cm[1][1]) / len(y_test),5) *100)
print('Performance KNN zoptymalizowanego przez GridSearch wynosi :', performance,'%')
