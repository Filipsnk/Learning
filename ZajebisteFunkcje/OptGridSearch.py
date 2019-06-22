######### Import bibliotek ##########

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


########## Dopasowanie algorytmu SVR wraz z ##########
########## petla optymalizujaca Grid Search ##########



n_estimators = [int(x) for x in np.linspace(1,1000,10, endpoint = False)]
n_neighbors = [int(x) for x in np.linspace(2,7,6, endpoint = False)]
max_depth = [int(x) for x in np.linspace(1,100,10, endpoint = False)]
n_neighbors = [2,3,4,5,6]
parameters = {'n_estimators' : n_estimators,
              'criterion' : ['gini','entropy'],
              'max_depth' : max_depth}

parameters_knn = {'n_neighbors': n_neighbors, 
                  'weights': ['uniform', 'distance'], 
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
######### Optymalizowane parametry #########


def OptimizedGridSearch(object_name, parameters_set, X_train, y_train, X_test, y_test): #!# dorzucic CV
    ## object_name, np. RandomForestClassifier
    ## parameters - slownik z parametrami do optymalizacji

    syntax = str(object_name)
#    code = compile(syntax, '<string>', 'exec')
#    exec(code)
    opt_object = object_name
    ######### Optymalizowane parametry - slownik #########
    
    parameters = parameters_set
    
    klucze = list(parameters.keys())
    wartosci = list(parameters.values())
    
    ######### Punkt startowy petli #########
    
    oceny = [] # lista z ocenami poszczegolnych modeli
    total_time = 0 
     
    opt_object.fit(X_train,y_train) 
    
    accuracies = cross_val_score(estimator = opt_object, X = X_test, y = y_test, cv = 3, n_jobs = -1) 
    oceny.append(str(0) + ': ' + 'mean: ' + str(round(accuracies.mean(),4)) + ' std: ' + str(round(accuracies.std(),4)))
    
        ##### Petla wlasciwa #####
    
    for i in range(0,len(parameters)):  
        parameters_set = {}
        parameters = klucze[i]
        values = wartosci [i]
    
        parameters_set[parameters] = values
        print(parameters_set)  
    
        ##### Pomiar czasu #####
    
        import time
        start = time.time()
    
        ##### Algorytm Grid Search #####
    
        
    
        grid_search = GridSearchCV(estimator = opt_object, 
                                   param_grid = parameters_set,
                                   cv = 5,
                                   n_jobs = -1)
    
        grid_search = grid_search.fit(X_train, y_train) 
    
        ##### Koniec pomiaru czasu #####
    
        end = time.time()
        exec_time = end-start
        total_time = total_time + exec_time 
    
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
    
        ##### Odczyt optymalnych parametrow #####
    
        for key, value in best_parameters.items():
            if isinstance(value, str):
                new_parameter = str(key) + '=\''  + str(value) + '\''
            else:
                new_parameter = str(key) + '='  + str(value)
    
        ##### Aktualizacja regressora #####
    
        if '()'  in syntax: ### regressor z opcja default 
            syntax = syntax.replace(')','') + str(new_parameter) + ')'  
        else:
            syntax = syntax.replace(')',',') + str(new_parameter) + ')'
        
        syntax_loop = compile(syntax, '<string>', 'exec')
        exec(syntax_loop) 
    
        opt_object.fit(X_train,y_train) 
    
        #### Model k-fold Cross Validation + zrzut wynikow #####
        accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 3, n_jobs = -1)
        oceny.append(str(i) + ': ' + 'mean: ' + str(round(accuracies.mean(),4)) + ' std: ' + str(round(accuracies.std(),4)))
    
        summary = 'Optymalny zestaw parametrow: ' + str(syntax) + '\n' + 'Czas trwania procedury:' + str(total_time)

    return summary


#OptimizedGridSearch('KNeighborsClassifier', parameters_knn)