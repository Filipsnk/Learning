from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys
sys.setrecursionlimit(10000)
######### Punkt startowy petli #########

oceny = [] # lista z ocenami poszczegolnych modeli
total_time = 0 


start = 'classifier = RandomForestClassifier(n_estimators = 20, max_depth = 10'
end = ')'


syntax = start + end
exec(syntax) 
print(syntax)
classifier.fit(X_train,y_train) 

accuracies = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10) 
oceny.append(str(0) + ': ' + 'mean: ' + str(round(accuracies.mean(),4)) + ' std: ' + str(round(accuracies.std(),4)))


    ##### Petla wlasciwa #####

for i in range(1,5):  
    
    parameters_set = {}
    
    parameters = klucze[i]
    values = wartosci [i]

    parameters_set[parameters] = values
    print(parameters_set)  

    ##### Pomiar czasu #####

    import time
    start = time.time()
    
    ##### Algorytm Grid Search #####
    
    grid_search = GridSearchCV(estimator = regressor, 
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
        new_parameter = str(key) + '=' + str(value)
    
    ##### Aktualizacja regressora #####
    
    syntax = syntax.replace(')',',') + str(new_parameter) + ')'
    exec(syntax) 
    regressor.fit(X_train,y_train) 
    
    #### Model k-fold Cross Validation + zrzut wynikow #####
    accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10, n_jobs = -1)
    oceny.append(str(i) + ': ' + 'mean: ' + str(round(accuracies.mean(),4)) + ' std: ' + str(round(accuracies.std(),4)))
    
oceny
total_time