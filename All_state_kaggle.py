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

### ENCODE 'STATE' VARIABLE END REMOVE IT FROM TRAINING SET ###

train_dummy = pd.concat([train,pd.get_dummies(train['state'])],axis=1)
del train_dummy['state']

### PRINT NULLS PERCENTAGE IN ALL COLUMNS ###

for i in train.columns:
    print(i,round(train[i].isnull().sum()/len(train[i]),2))

### FILL NULLS IN RISK FACTOR COLUMN ###

#NIE POZWOLIMY NA TO#





