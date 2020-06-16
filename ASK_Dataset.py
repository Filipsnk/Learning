### Define paths for proper runtime from beginning to end 

modules_path_m = 'C://Users//Marek//Desktop//Python//FilipSNK'
datasets_path_m = 'C://Users//Marek//Desktop//Python//Kaggle//AllState'

modules_path_m1 = '//Users//marekpytka//Documents//Programowanie//Learning'
datasets_path_m1 = '//Users//marekpytka//Documents//Programowanie//AllState'

#modules_path_f
#datasets_path_f

import pandas as pd, numpy as np, pandasql as sql, webbrowser as wbr, seaborn as sns
import os, datetime

os.chdir(modules_path_m)

from flpmarlib import *
#!#from flpmarlib import (null_summary, heatmap_and_corr, time_of_day, is_weekend,
#                       map_us_region, df_cleanup, reduce_memory_usage, remove_objects)

### KAGGLE + NOTES LINK ###
wbr.open_new_tab('https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data')
wbr.open_new_tab('https://docs.google.com/document/d/1c82cgdvH0DUGtxy-5OWSMhLfqCo5GkRDTo6OBk7NvwQ/edit')
wbr.open_new_tab('https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74')

### SET WORKING DIRECTORY ###
os.chdir(datasets_path_m)

### IMPORT DATA ###
train = pd.read_csv('train.csv')
test = pd.read_csv('test_v2.csv')
sample_submission = pd.read_csv('sampleSubmission.csv')

### PRINT NULLS PERCENTAGE IN ALL COLUMNS FOR TRAIN AND TEST ###

null_summary(train)
null_summary(test)

# Można dodać rozkłady wieku dla każdego A

### CHECK CORELATION BETWEEN RISK_FACTOR AND OTHER VARIABLES - HEATMAP ###
    
heatmap_and_corr(train, 'risk_factor', 10)

### FEATURE ENGINEERING ###

### NO_OF_OFFERS - Number of offers generated for each customer

columns = ['shopping_pt', 'customer_ID']
offers_groupby = train[columns].groupby('customer_ID').count()
offers_groupby.columns = ['no_of_offers']

train = train.merge(offers_groupby, left_on = 'customer_ID', 
                    right_on = 'customer_ID', how = 'left')

offers_groupby = test[columns].groupby('customer_ID').count()
offers_groupby.columns = ['no_of_offers']

test = test.merge(offers_groupby, left_on = 'customer_ID', 
                    right_on = 'customer_ID', how = 'left')

### AVG_COST - average cost of one offer

columns = ['cost', 'customer_ID']
costs_groupby = train[columns].groupby('customer_ID').mean()
costs_groupby.columns = ['avg_cost']

train = train.merge(costs_groupby, left_on = 'customer_ID', 
                    right_on = 'customer_ID', how = 'left')

costs_groupby = test[columns].groupby('customer_ID').mean()
costs_groupby.columns = ['avg_cost']

test = test.merge(costs_groupby, left_on = 'customer_ID', 
                    right_on = 'customer_ID', how = 'left')

### TIME OF DAY (0 - morning, 1 - afternoon, 2- evening)

# Extract hour from Time variable and convert to int

train['hour'] = train['time'].str.slice(start=0, stop = 2).astype('int')  
test['hour'] = test['time'].str.slice(start=0, stop = 2).astype('int')

train['time_of_day'] = train.apply(time_of_day, axis=1)
test['time_of_day'] = test.apply(time_of_day, axis=1)

### IsWeekend 0 - weekday, 1 - weekend

train['is_weekend'] = train.apply(is_weekend, axis=1)
test['is_weekend'] = test.apply(is_weekend, axis=1)

### GROUP AGE DIFFERENCE (age_oldest - age_youngest)

train['age_diff'] = train['age_oldest'] - train['age_youngest'] 
test['age_diff'] = test['age_oldest'] - test['age_youngest'] 

### MAP CAR_VALUE VARIABLE ### ##!!## needs one-hot

car_value_dict = dict({'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8,
                      'i':9})

train['car_value_map'] = train['car_value'].map(car_value_dict)
test['car_value_map'] = test['car_value'].map(car_value_dict)

### MAP 'STATE' VARIABLE END REMOVE IT FROM TRAINING SET ###

map_us_region(train, 'state')
map_us_region(test, 'state')

### SELECT ACCEPTED_OFFERS FOR TRAINING ONLY
accepted_offers = train.loc[train['record_type']==1,:]
accepted_offers_test = test.loc[test['record_type']==1,:]

### BUILD DUMMIES FROM US_REGION, RISK_FACTOR, CAR_VALUE

columns = ['US_Region', 'risk_factor', 'car_value_map']
accepted_train_dummy = dummify_dataset(accepted_offers, columns)
accepted_test_dummy = dummify_dataset(accepted_offers_test, columns)

### CLEANUP OF UNNECESSARY COLUMNS

deleted_columns = ['car_value', 'day', 'hour', 'time', 'state', 'shopping_pt',
                   'record_type', 'location', 'US_Region', 'risk_factor',
                   'car_value_map']

df_cleanup(accepted_train_dummy, deleted_columns)
df_cleanup(accepted_test_dummy, deleted_columns)

### CLEANUP OF UNNECESSARY OBJECTS

objects = ['costs_groupby', 'offers_groupby']
remove_objects(objects)

### FILL NANs (NAIVE APPROACH - USE MEDIAN))

null_dict = {'risk_factor': train['risk_factor'].median(), 
          'C_previous': train['C_previous'].median(), 
          'car_value_map': train['car_value_map'].median(), 
          'duration_previous': train['duration_previous'].median()}

accepted_train_dummy = accepted_train_dummy.fillna(value=null_dict)
accepted_test_dummy = accepted_test_dummy.fillna(value=null_dict)

### CLEANUP OF MEMORY USAGE

reduce_memory_usage(accepted_train_dummy, verbose=True)
reduce_memory_usage(accepted_test_dummy, verbose=True) 

# Clean accepted_train_dummy from columns that we'd like to predict 

ordered_columns = ['C', 'A', 'E', 'D', 'B', 'F', 'G'] ## ordered desceding via 

df_cleanup(accepted_train_dummy, ordered_columns)
df_cleanup(accepted_test_dummy, ordered_columns)

# Set index of customer_ID 
    
accepted_train_dummy = accepted_train_dummy.set_index(['customer_ID'])
accepted_test_dummy = accepted_test_dummy.set_index(['customer_ID'])

accepted_train_dummy.to_csv('accepted_train_dummy.csv')
accepted_test_dummy.to_csv('accepted_test_dummy.csv')

###############################################################################
### NOT USED ###
###############################################################################


### PRZESZCZEP ###

### Preparing data for k-means clustering

new_data['age_diff'] = new_data['age_oldest'] - new_data['age_youngest']
new_data['option'] = new_data[['A','B','C','D','E','F','G']].astype(str).apply(lambda x: ''.join(x.astype(str)),axis=1)
new_data['option'] = new_data['option'].astype(int)

### Define location

west = ['WA','OR','ID','NV','MT','WY','UT','CO','NM']
new_data['is_west'] = new_data['state'].apply(lambda x: 1 if x in west else 0)

# Histogram of option
   
plt.hist(new_data['option'],bins=50)
plt.show()

## Create quartiles for option_2 based on the above plot

new_data['option_q'] = pd.qcut(new_data['option'],3,labels=[1,2,3])

# For each option from the above (1,2,3) assign risk factor

new_data.groupby(['option_q'])['risk_factor'].value_counts()

# For each option_q and select the most frequent risk_factor

risk_mode = new_data.groupby(['option_q']).apply(lambda x: x['risk_factor'].value_counts().index[0]).to_dict()
final_data = new_data.copy()

# Fill in missing values based on the above.
for a,b in enumerate(final_data['risk_factor'].isnull()):
    if b == True:
        final_data.iloc[a,12]=risk_mode[final_data.iloc[a,29]]

# How many missing values

for i in final_data.columns:
    print('There is {} missing values in column {}'.format(final_data[i].isnull().sum(),i))

# Input value
    
final_data = final_data.fillna(method ='pad')

### Optional ### 
### MinMax to standarize data

scaler=MinMaxScaler()

### Columns for clustering

columns = ['cost','car_value_2']
clustered_data = new_data[columns]
clustered_data = scaler.fit_transform(clustered_data)

### Plot of data
plt.scatter(clustered_data[:,1],clustered_data[:,0])

### KONIEC PRZESZCZEPU ###

