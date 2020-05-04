#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, pandasql as sql, webbrowser as wbr, seaborn as sns
import os, datetime

### KAGGLE + NOTES LINK ###
wbr.open_new_tab('https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data')
wbr.open_new_tab('https://docs.google.com/document/d/1c82cgdvH0DUGtxy-5OWSMhLfqCo5GkRDTo6OBk7NvwQ/edit')

### SET WORKING DIRECTORY ###
os.chdir('C://Users//Marek//Desktop//Python//Kaggle//AllState')

### IMPORT DATA ###
train = pd.read_csv('train.csv')
test = pd.read_csv('test_v2.csv')
sample_submission = pd.read_csv('sampleSubmission.csv')

### PRINT NULLS PERCENTAGE IN ALL COLUMNS FOR TRAIN AND TEST ###

def null_summary(dataset):

    for i in dataset.columns:
        print(i,round(dataset[i].isnull().sum()/len(dataset[i]),2)*100)

null_summary(train)
null_summary(test)

### SQL CHECKS ###

### CHECK OF RANDOM CUSTOMERS ### 

query = 'select * from train where Customer_ID in (10015208,10015221,10015226,10015288)'
customer_data = sql.sqldf(query, locals())

### CHECK IF THE NUMBER OF STATES IS EQUAL BETWEEN TRAIN AND TEST ###

query_1 = 'SELECT DISTINCT state FROM train'
query_2 = 'SELECT DISTINCT state FROM test'

distinct_states_train = sql.sqldf(query_1, locals())
distinct_states_test = sql.sqldf(query_2, locals())

check_1 = 'SELECT state FROM train WHERE state NOT IN (SELECT state FROM test)'
check_1_result = sql.sqldf(check_1, locals())

### CHECK HOW MANY NULLS ARE THERE IN RISK_FACTOR ###

risk_nulls = 'SELECT COUNT(*) FROM train WHERE risk_factor is NULL'
risk_nulls_result = sql.sqldf(risk_nulls, locals())

### CHECK DEPENDENCY BETWEEN RISK_FACTOR AND AGE_OLDEST / YOUNGEST ###

risk_factor_nulls = 'SELECT risk_factor, MAX(age_oldest), MIN(age_oldest), MAX(age_youngest), MIN(age_youngest) FROM train'
risk_factor_nulls_2 = 'SELECT risk_factor, AVG(age_oldest), AVG(age_youngest) FROM train GROUP BY risk_factor'
risk_factor_nulls_results = sql.sqldf(risk_factor_nulls, locals())
risk_factor_nulls_2_results = sql.sqldf(risk_factor_nulls_2, locals())

### CHECK A DEPENDENCY ON CAR_AGE ###

option_A = 'SELECT A, AVG(car_age) FROM train GROUP BY A'
option_A_results = sql.sqldf(option_A, locals())

option_A_age = 'SELECT A, AVG(age_oldest), AVG(age_youngest) FROM train GROUP BY A'
option_A_age_results = sql.sqldf(option_A_age, locals())

### CHECK MINIMUM AND MAXIMUM AGE ###

hour_check = 'SELECT MAX(hour), MIN(hour) FROM train'
print(sql.sqldf(hour_check, locals()))

### CAR_VALUE VARIABLE VALUES ###
car_value_check = 'SELECT DISTINCT car_value FROM train'
print(sql.sqldf(car_value_check, locals()))

### CODE FOR GROUPPING STATES (NORTH/SOUTH/EAST/WEST) ###

state_check = 'SELECT DISTINCT state FROM train'
print(sql.sqldf(state_check, locals()))

# Można dodać rozkłady wieku dla każdego A

### CHECK CORELATION BETWEEN RISK_FACTOR AND OTHER VARIABLES - HEATMAP ###

def heatmap_and_corr(dataset, column, top_correlations):
    
    corr = dataset.corr()
    
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    
    print(abs(corr[column]).sort_values(ascending = False)[1:top_correlations + 1])
    
heatmap_and_corr(train, 'risk_factor', 10)

### FEATURE ENGINEERING ###

### TIME OF DAY (0 - morning, 1 - afternoon, 2- evening)

# Extract hour from Time variable and convert to int
train['hour'] = train['time'].str.slice(start=0, stop = 2).astype('int')  
test['hour'] = test['time'].str.slice(start=0, stop = 2).astype('int')

def time_of_day(row):
    if row['hour'] >= 6 and row['hour'] <= 11:
        val = 0
    elif row['hour'] >= 12 and row['hour'] <= 19:
        val = 1
    else:
        val = 2
    return val

train['time_of_day'] = train.apply(time_of_day, axis=1)
test['time_of_day'] = test.apply(time_of_day, axis=1)

### GROUP AGE DIFFERENCE (age_oldest - age_youngest)

train['age_diff'] = train['age_oldest'] - train['age_youngest'] 
test['age_diff'] = test['age_oldest'] - test['age_youngest'] 

### MAP CAR_VALUE VARIABLE ###

car_value_dict = dict({'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8,
                      'i':9})

train['car_value_map'] = train['car_value'].map(car_value_dict)
test['car_value_map'] = test['car_value'].map(car_value_dict)

del train['car_value']
del test['car_value']


### MAP 'STATE' VARIABLE END REMOVE IT FROM TRAINING SET ###


regions= {'West': ['WA', 'OR', 'CA', 'NV', 'ID', 'UT'],
                  'North': ['MT', 'WY', 'ND', 'SD', 'NE', 'MN', 'IA', 'WI', 
                            'IL', 'MI', 'IN', 'OH', 'PA', 'VT', 'MO',
                            'KS', 'WI'],
                  'South': ['AZ', 'NM', 'TX', 'OK', 'AR', 'LA', 
                            'TN', 'MS', 'AL', 'GA', 'FL', 'CO'],
                  'East': ['SC', 'NC', 'VA', 'WV', 'DC', 'MD', 'DE', 'NJ', 
                           'CT', 'RI', 'MA', 'NH', 'ME', 'NY', 'ME', 'KY']}

region_mapping = {}

for keys, values in regions.items():
    for value in values:
        region_mapping[value] = keys

train['US_Region'] = train['state'].map(region_mapping)
test['US_Region'] = test['state'].map(region_mapping)

del train['state']
del test['state']

train_dummy = pd.concat([train,pd.get_dummies(train['US_Region'])],axis=1)

### SELECT ACCEPTED_OFFERS FOR TRAINING ONLY
accepted_offers = train.loc[train['record_type']==1,:]
accepted_train_dummy = pd.concat([accepted_offers,
                                  pd.get_dummies(accepted_offers['US_Region'])],axis=1)
    
del train_dummy['time']
del accepted_train_dummy['time']

heatmap_and_corr(train_dummy, 'US_Region', 10)

### FILL NANs (NAIVE APPROACH - USE MEAN/MEDIAN))

values = {'risk_factor': train['risk_factor'].median(), 
          'C_previous': train['C_previous'].median(), 
          'car_value_map': train['car_value_map'].median(), 
          'duration_previous': train['duration_previous'].median()}

train_dummy = train_dummy.fillna(value=values)
accepted_train_dummy = accepted_train_dummy.fillna(value=values)
test_dummy = test.fillna(value=values)
accepted_test_dummy = test_dummy.fillna(value=values)

### RANDOM FOREST ATTEMPT ###

from sklearn.preprocessing import StandardScaler

# Removing unnecessary variables #

# Splitting the dataset into the Training set and Test set
pred_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
y_full = accepted_train_dummy[pred_columns]

for column in pred_columns:
    del accepted_train_dummy[column]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

full_predictions=pd.DataFrame(columns = pred_columns)

### Fitting Random Forest Classifier to each Insurance Option ###
    
accuracy_total = []
iteration_count = 1

started_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Code execution started at: ' + started_time)

for column in pred_columns:
    
    print('Iteration number ' + str(iteration_count) + ' has started for column: ' + column)
    print('Start time: ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    y = y_full[column]
    X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print ('Fitting the RFC classifier for column: ' + column)
    
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
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

sum(accuracy_total)/len(accuracy_total)
## 30 -> 67.04
## 50 -> 67.39
## 100 -> 67.86
null_summary(accepted_train_dummy)    
full_predictions.dtypes

full_predictions_str = full_predictions.astype('str')
full_predictions_str['total'] = full_predictions_str[pred_columns].agg(''.join, axis=1)

y_full_str = y_full.iloc[:166313,:].astype('str')
y_full_str['total'] = y_full_str[pred_columns].agg(''.join, axis=1)

i=0
count_correct = 0

for i in range(len(y_full_str)):
    if y_full_str['total'][i] == full_predictions_str['total'][i]:
        count_correct +=1
    else:
        if i % 100 == 0:
            print('Iteration: ' + str(i))
print(count_correct, ' correct predictions out of ', len(y_full_str))

### FILL NULLS IN RISK FACTOR COLUMN ###

###############################################################################
### NOT USED ###
###############################################################################


### CHECK NULLS BEFORE RUNNING RANDOM FOREST CLASSIFIER ###
for i in X_train.columns:
    print(str(i) +' ' +str(round(X_train[i].isnull().sum())))

array_sum = np.sum(X_train)
array_has_nan = np.isnan(array_sum)

