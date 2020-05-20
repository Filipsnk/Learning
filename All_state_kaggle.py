## CODE CHANGE 

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

def time_of_day(row): ##!!## needs one-hot encode
    if row['hour'] >= 6 and row['hour'] <= 11:
        val = 0
    elif row['hour'] >= 12 and row['hour'] <= 19:
        val = 1
    else:
        val = 2
    return val

train['time_of_day'] = train.apply(time_of_day, axis=1)
test['time_of_day'] = test.apply(time_of_day, axis=1)

del train['hour']
del test['hour']

del train['time']
del test['time']

### IsWeekend 0 - weekday, 1 - weekend

def is_weekend(row): ##needs one-hot encode
    if row['day'] >= 0 and row['day'] <= 4:
        val = 0
    else:
        val = 1
    return val

train['is_weekend'] = train.apply(is_weekend, axis=1)
test['is_weekend'] = test.apply(is_weekend, axis=1)

del train['day']
del test['day']

### GROUP AGE DIFFERENCE (age_oldest - age_youngest)

train['age_diff'] = train['age_oldest'] - train['age_youngest'] 
test['age_diff'] = test['age_oldest'] - test['age_youngest'] 

### MAP CAR_VALUE VARIABLE ### ##!!## needs one-hot

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

del train['location'] #not necessary
del test['location']

### SELECT ACCEPTED_OFFERS FOR TRAINING ONLY
accepted_offers = train.loc[train['record_type']==1,:]
accepted_train_dummy = pd.concat([accepted_offers,
                                  pd.get_dummies(accepted_offers['US_Region'])],axis=1)

del accepted_train_dummy['US_Region']

### FILL NANs (NAIVE APPROACH - USE MEDIAN))

null_dict = {'risk_factor': train['risk_factor'].median(), 
          'C_previous': train['C_previous'].median(), 
          'car_value_map': train['car_value_map'].median(), 
          'duration_previous': train['duration_previous'].median()}

accepted_train_dummy = accepted_train_dummy.fillna(value=null_dict)
test_dummy = test.fillna(value=null_dict)
accepted_test_dummy = test_dummy.fillna(value=null_dict)

### RANDOM FOREST ATTEMPT ###

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Preparation of prediction dataframe, where output of each prediction vector
# will be appended to

pred_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
y_full = accepted_train_dummy[pred_columns] # our equivalent of y_test

# Clean accepted_train_dummy from columns that we'd like to predict 

for column in pred_columns:
    del accepted_train_dummy[column] 

# Set index of customer_ID 
    
accepted_train_dummy = accepted_train_dummy.set_index(['customer_ID'])
full_predictions=pd.DataFrame(columns = pred_columns) # our equivalent of y_pred

### Fitting Random Forest Classifier to each Insurance Option ###

mode = input('Choose 0 for standardization and 1 for normalization: ')
no_of_trees = int(input('Choose how many trees should be built within RFC model:'))    
accuracy_total = []
iteration_count = 1

started_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('Code execution started at: ' + started_time)

for column in pred_columns:
    
    print('Start time: ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    y = y_full[column]
    X_train, X_test, y_train, y_test = train_test_split(accepted_train_dummy, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    if mode == 0:
        sc = StandardScaler()
    else:
        sc = MinMaxScaler()
        
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print ('Fitting the RFC classifier for column: ' + column)
    
    classifier = RandomForestClassifier(n_estimators = no_of_trees, criterion = 'entropy', random_state = 0)
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
                      
sum(accuracy_total)/len(accuracy_total)

# Convert full_predictions so that the values for each insurance option can be aggregated to one string

full_predictions_str = full_predictions.astype('str')
full_predictions_str['total'] = full_predictions_str[pred_columns].agg(''.join, axis=1)

y_full_str = y_full.astype('str')
y_full_str['total'] = y_full_str[pred_columns].agg(''.join, axis=1)

y_full_str = y_full_str.reset_index()

# Calculate number correct predictions 

i=0
count_correct = 0

for i in range(len(y_full_str)): ##!!## sprawdzic czemu nie dziala
    if y_full_str['total'][i] == full_predictions_str['total'][i]:
        count_correct +=1
#    else:
#        if i % 100 == 0:
#            print('Iteration: ' + str(i))
print(count_correct, ' correct predictions out of ', len(y_full_str))

### FILL NULLS IN RISK FACTOR COLUMN ###

###############################################################################
### NOT USED ###
###############################################################################
