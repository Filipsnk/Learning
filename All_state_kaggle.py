#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd, numpy as np, pandasql as sql, webbrowser as wbr
import os

wbr.open_new_tab('https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data')

os.chdir('C://Users//Marek//Desktop//Python//Kaggle//AllState')

### IMPORT DATA ###

train = pd.read_csv('train.csv')
test = pd.read_csv('test_v2.csv')
sample_submission = pd.read_csv('sampleSubmission.csv')

query = 'select * from train where Customer_ID in (10015208,10015208,10015221,10015226,10015288)'
customer_data = sql.sqldf(query, locals())

### CHECK IF THE NUMBER OF STATES IS EQUAL BETWEEN TRAIN AND TEST ###

query_1 = 'SELECT DISTINCT state FROM train'
query_2 = 'SELECT DISTINCT state FROM test'

distinct_states_train = sql.sqldf(query_1, locals())
distinct_states_test = sql.sqldf(query_2, locals())

check_1 = 'SELECT state FROM train WHERE state NOT IN (SELECT state FROM test)'
check_1_result = sql.sqldf(check_1, locals())

### ENCODE 'STATE' VARIABLE END REMOVE IT FROM TRAINING SET ###

train_dummy = pd.concat([train,pd.get_dummies(train['state'])],axis=1)
del train_dummy['state']

### PRINT NULLS PERCENTAGE IN ALL COLUMNS ###

for i in train.columns:
    print(i,round(train[i].isnull().sum()/len(train[i]),2))


### FEATURE ENGINEERING ###

### TIME OF DAY (0 - morning, 1 - afternoon, 2- evening)

# Extract hour from Time variable and convert to int
train_dummy['hour'] = train_dummy['time'].str.slice(start=0, stop = 2).astype('int') 

def time_of_day(row):
    if row['hour'] >= 6 and row['hour'] <= 11:
        val = 0
    elif row['hour'] >= 12 and row['hour'] <= 19:
        val = 1
    else:
        val = 2
    return val

train_dummy['time_of_day'] = train_dummy.apply(time_of_day, axis=1)

hour_check = 'SELECT MAX(hour), MIN(hour) FROM train_dummy'
print(sql.sqldf(hour_check, locals()))


### FILL NULLS IN RISK FACTOR COLUMN ###

###############################################################################
### NOT USED ###
###############################################################################

# IS ACCEPTED (1 - offer has been accepted, 0 - not accepted yet)

query_ao= 'SELECT customer_ID, MAX(shopping_pt) AS shopping_pt, 1 As IsAccepted FROM train GROUP BY customer_id'
accepted_offers = sql.sqldf(query_ao, locals())

train_dummy = train_dummy.merge(accepted_offers,
                              indicator=True, # zakomentowac, zeby nie pojawialo sie _merge
                              how='left',
                              left_on=["customer_ID", "shopping_pt"],
                              right_on=["customer_ID", "shopping_pt"])

del train_dummy['_merge']
train_dummy['IsAccepted'] = train_dummy['IsAccepted'].fillna(0)

  


