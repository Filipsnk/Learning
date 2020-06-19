### Import dataset from ASK_Dataset file

import pandas as pd, numpy as np, webbrowser as wbr
import os

modules_path_m = 'C://Users//Marek//Desktop//Python//FilipSNK'
datasets_path_m = 'C://Users//Marek//Desktop//Python//Kaggle//AllState'

os.chdir(datasets_path_m)

accepted_train_dummy = pd.read_csv('accepted_train_dummy.csv')
accepted_test_dummy = pd.read_csv('accepted_test_dummy.csv')

### RANDOM FOREST ATTEMPT ###

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

### Get Random Forest Classifier params to run Grid Search tuning ###

rf = RandomForestClassifier(random_state = 42)

print('Parameters available for grid-search optimization:')
print(rf.get_params())

### Reference website for grid creation

wbr.open_new_tab('https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74')
from sklearn.model_selection import RandomizedSearchCV


### Creating grid of possible parameters' value ### 

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 300, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None) 
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

### Present the grid of possible values:

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Preparation of prediction dataframe, where output of each prediction vector
# will be appended to

pred_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
ordered_columns = ['C', 'A', 'E', 'D', 'B', 'F', 'G'] ## ordered desceding via 
                                                      ## accuracy in first turn

y_full = accepted_train_dummy[pred_columns] # our equivalent of y_test
full_predictions=pd.DataFrame(columns = pred_columns) # our equivalent of y_pred

# Clean accepted_train_dummy from columns that we'd like to predict 

df_cleanup(accepted_train_dummy, ordered_columns)
df_cleanup(accepted_test_dummy, ordered_columns)

### Prepare dataset for modelling ###
### Fitting Random Forest Classifier to each Insurance Option ###

mode = input('Choose 0 for standardization and 1 for normalization: ')
grid_search = input('Choose if you want to tune algorithm via grid search (0 - no, 1 - yes): ')

if grid_search == '0':
    no_of_trees = int(input('Choose how many trees should be built within RFC model: '))  
  
accuracy_total = []
iteration_count = 1
iterations_rf = {}
started_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Code execution started at: ' + started_time)

for column in ordered_columns:
    
    print('Start time: ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    y = y_full[column]
    X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy, 
                                                        y, test_size = 0.25, 
                                                        random_state = 0)

    # Feature Scaling
    if mode == 0:
        sc = StandardScaler()
    else:
        sc = MinMaxScaler()
        
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print ('Fitting the RFC classifier for column: ' + column)
  
    if grid_search == '0':    
        classifier = RandomForestClassifier(n_estimators = no_of_trees, 
                                            criterion = 'entropy', 
                                            random_state = 0)
        classifier.fit(X_train, y_train)

    elif grid_search == '1': 
        classifier = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                    n_iter = 100, cv = 3, verbose=2, 
                                    random_state=42, n_jobs = -1)
        classifier.fit(X_train, y_train)    

    y_pred = classifier.predict(X_test)
     
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    print ('Confusion matrix for column: ' + column)
    print(cm)

    hits = 0
    
    for i in range(len(cm)):
        hits = hits + cm[i,i]
    
    attempts = sum(sum(cm))    
    accuracy = round((hits/attempts*100),2)
    accuracy_total.append(accuracy)
    
    
    print('Accuracy: ', accuracy, '%')
    print('End time: ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # Predicting total results
    y_pred_total = classifier.predict(accepted_train_dummy)
    full_predictions[column] = y_pred_total
    
    # Append new predictions and dummify them
    
    accepted_train_dummy[column] = y_pred_total
    accepted_train_dummy = pd.concat([accepted_train_dummy,
                                  pd.get_dummies(accepted_train_dummy[column])],axis=1)
    # Append accuracy per column
       
    iterations_rf[column] = accuracy
    
    # Append the predictions to accepted_train_dummy (X_train)
    
sum(accuracy_total)/len(accuracy_total)

# Convert full_predictions so that the values for each insurance option can be aggregated to one string

full_predictions_str = full_predictions.astype('str')
full_predictions_str['total'] = full_predictions_str[pred_columns].agg(''.join, axis=1)

y_full_str = y_full.astype('str')
y_full_str['total'] = y_full_str[pred_columns].agg(''.join, axis=1)
y_full_str = y_full_str.reset_index()


i=0
count_correct = 0

for i in range(len(y_full_str)): 
    if y_full_str['total'][i] == full_predictions_str['total'][i]:
        count_correct +=1

print(count_correct, ' correct predictions out of ', len(y_full_str))
  