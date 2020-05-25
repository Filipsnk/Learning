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

<<<<<<< HEAD

## Dla kazdego klienta wyciagnij jego finalna polise ktora kupil (ostatni wiersz per klient)
def klient(df):
        
    klient_zakup= pd.DataFrame()
    
    Customers=list(set(df['customer_ID']))
    
    for i in Customers:
        
        cust=df.loc[df['customer_ID']==i,:][-1:]
        
        klient_zakup = klient_zakup.append(cust)
        print('Dodawanie klienta numer: ',i)


    
=======
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
>>>>>>> 25f53fdb4d28fa55c9b84a36b0e2a3cd59b358c9

  
========= DATA PREP   

## Functions

def detect_outliers(df,columns):
    
    clean_df = df
    
    try:
    
        for col in columns:
                        
            first_quartile = clean_df[col].quantile(0.25)
            third_quartile = clean_df[col].quantile(0.75)
            
            IQR = third_quartile-first_quartile
            
            indx = clean_df[(clean_df[col] < (third_quartile+IQR*1.5))  &  (clean_df[col]>(first_quartile-IQR*1.5))].index
    
            diff = clean_df.shape[0]-indx.shape[0]
    
            print('In column: {} there are {} observations to be deleted'.format(col,diff))
            
            clean_df = clean_df.iloc[indx].copy().reset_index()
            
            del clean_df['index']
            
        return clean_df
    
    except:
        print('Check data types of columns')
        
        

def elbow_method (df,no_clusters):


    distance = []
    
    for i in range(1,no_clusters):
        
        km = KMeans(n_clusters = i
                    ,init='random'
                    ,n_init=10
                    ,max_iter=300
                    ,random_state =0)
        
        model = km.fit(df)
        
        distance.append(km.inertia_)
    
    plt.plot(range(1,no_clusters),distance,marker = 'o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distances')
    plt.show()            



def check_distribution (data,columns):
            
    for i in data[columns].columns:
            
        print('\t\t\nHistogram for column: {}'.format(i))
            
        data[i].hist(bins=50)
        
        plt.show()


# get data
data = pd.read_csv('train.csv')


#Select the last row for each customer. When customer boughted final product
new_data = data.loc[data['record_type']==1,:].reset_index()

# How many missing values

for i in new_data.columns:
    
    print('There is {} missing values in column {}'.format(new_data[i].isnull().sum(),i))

# correlations

corr = new_data.corr()


# Sum costs per customer as attribute for clustering
spend = pd.DataFrame(data.groupby(['customer_ID'])['cost'].mean().reset_index())

# Combine two datasets. Optional

#new_data = new_data.merge(spend,on='customer_ID')

### Analysis of costs in order to define car value

median= new_data['cost'].median()
new_data['cost_label'] = new_data['cost'].apply(lambda x: 1 if x > median else 0)

cost_above = new_data.loc[new_data['cost_label']==1 ,['car_value']]
cost_below = new_data.loc[new_data['cost_label']==0 ,['car_value']]


y_1 = [y_1 for _,y_1 in enumerate(cost_above['car_value'].value_counts().sort_index())]
x_1 = np.arange(1,10)
xticks = cost_above['car_value'].value_counts().sort_index().keys().tolist()

y_2 = [y_2 for _,y_2 in enumerate(cost_below['car_value'].value_counts().sort_index())]
x_2 = np.arange(1,10)


fig = plt.figure()
fig, axs = plt.subplots(1,1,figsize=(10,8))
plt.bar(x_1,y_1,width=0.5, color='b', linewidth=3,label='Above_median')
plt.bar(x_2,y_2,width=0.2, color='r', linewidth=0,label='Below_median')
plt.xticks(range(1,len(xticks)),xticks)
plt.xlabel('car_value')
plt.ylabel('count')
plt.legend()
plt.show()


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

