### Import dataset from ASK_Dataset file

import pandas as pd
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

# Preparation of prediction dataframe, where output of each prediction vector
# will be appended to

pred_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
ordered_columns = ['C', 'A', 'E', 'D', 'B', 'F', 'G'] ## ordered desceding via 
                                                      ## accuracy in first turn

y_full = accepted_train_dummy[pred_columns] # our equivalent of y_test

### Prepare dataset for modelling ###

full_predictions=pd.DataFrame(columns = pred_columns) # our equivalent of y_pred

### Fitting Random Forest Classifier to each Insurance Option ###

mode = input('Choose 0 for standardization and 1 for normalization: ')
no_of_trees = int(input('Choose how many trees should be built within RFC model:'))    
accuracy_total = []
iteration_count = 1
iterations_rf = {}
started_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Code execution started at: ' + started_time)

for column in ordered_columns:
    
    print('Start time: ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    y = y_full[column]
    X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy_raw, 
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
    
    classifier = RandomForestClassifier(n_estimators = no_of_trees, 
                                        criterion = 'entropy', 
                                        random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
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
  