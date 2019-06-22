######### Import bibliotek ##########

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from ZajebisteFunkcje import OptGridSearch
########## Dopasowanie algorytmu SVR wraz z ##########
########## petla optymalizujaca Grid Search ##########

start = 'classifier = RandomForestClassifier('
end = ')'

######### Optymalizowane parametry #########

n_estimators = [int(x) for x in np.linspace(1,1000,10, endpoint = False)]
max_depth = [int(x) for x in np.linspace(1,100,10, endpoint = False)]

parameters = {'n_estimators' : n_estimators,
              'criterion' : ['gini','entropy'],
              'max_depth' : max_depth}

######### Optymalizowane parametry - slownik #########

klucze = list(parameters.keys())
wartosci = list(parameters.values())

######### Punkt startowy petli #########

oceny = [] # lista z ocenami poszczegolnych modeli
total_time = 0 

syntax = start + end
exec(syntax) 
classifier.fit(X_train,y_train) 

accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 3, n_jobs = -1) 
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

    

    grid_search = GridSearchCV(estimator = classifier, 
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
    
    syntax
    exec(syntax) 

    classifier.fit(X_train,y_train) 

    #### Model k-fold Cross Validation + zrzut wynikow #####
    accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 3, n_jobs = -1)
    oceny.append(str(i) + ': ' + 'mean: ' + str(round(accuracies.mean(),4)) + ' std: ' + str(round(accuracies.std(),4)))

oceny
total_time
syntax