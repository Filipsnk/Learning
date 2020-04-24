#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, pandasql as sql, webbrowser as wbr, seaborn as sns
import os

### KAGGLE + NOTES LINK ###
wbr.open_new_tab('https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data')
wbr.open_new_tab('https://docs.google.com/document/d/1c82cgdvH0DUGtxy-5OWSMhLfqCo5GkRDTo6OBk7NvwQ/edit')

### SET WORKING DIRECTORY ###
os.chdir('C://Users//Marek//Desktop//Python//Kaggle//AllState')

### IMPORT DATA ###

train = pd.read_csv('train.csv')
test = pd.read_csv('test_v2.csv')
sample_submission = pd.read_csv('sampleSubmission.csv')

### ACCEPTED OFFERS ###
accepted_offers = train.loc[train['record_type']==1,:]

### PRINT NULLS PERCENTAGE IN ALL COLUMNS ###

for i in train.columns:
    print(i,round(train[i].isnull().sum()/len(train[i]),2)*100)
    
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
# Można dodać rozkłady wieku dla każdego A

### CHECK CORELATION BETWEEN RISK_FACTOR AND OTHER VARIABLES - HEATMAP ###

corr = train.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

abs(corr['risk_factor']).sort_values(ascending = False)[1:6]


### FEATURE ENGINEERING ###

### TIME OF DAY (0 - morning, 1 - afternoon, 2- evening)

# Extract hour from Time variable and convert to int
train['hour'] = train['time'].str.slice(start=0, stop = 2).astype('int')  

def time_of_day(row):
    if row['hour'] >= 6 and row['hour'] <= 11:
        val = 0
    elif row['hour'] >= 12 and row['hour'] <= 19:
        val = 1
    else:
        val = 2
    return val

train['time_of_day'] = train.apply(time_of_day, axis=1)

### GROUP AGE DIFFERENCE (age_oldest - age_youngest)

train['age_diff'] = train['age_oldest'] - train['age_youngest'] 

### ENCODE 'STATE' VARIABLE END REMOVE IT FROM TRAINING SET ###

train_dummy = pd.concat([train,pd.get_dummies(train['state'])],axis=1)
del train_dummy['state']

# Porobić grupowanie stanów 

### RANDOM FOREST ATTEMPT ###

from sklearn.preprocessing import StandardScaler

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.transform(test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

### FILL NULLS IN RISK FACTOR COLUMN ###

###############################################################################
### NOT USED ###
###############################################################################

    
'SELECT * FROM train INNER JOIN' +
'(SELECT customer_ID, MAX(shopping_pt) FROM train GROUP BY customer_ID) AS Max_Order'  +
'ON train.customer_ID = Max_Order.customerID AND train.shopping_pt = Max_Order.shopping_pt'


  
