### Library of functions done by Filip & Marek for Kaggle competitions ###

### Function takes dataset and its columns as input and returns histogram 
### to provide insight for its distribution

def check_distribution (data,columns):
    
    import matplotlib.pyplot as plt 
    
    for i in data[columns].columns:            
        print('\t\t\nHistogram for column: {}'.format(i))            
        data[i].hist(bins=50)        
        plt.show()

### Function takes dataset as an input and returns information about number of 
### outliers as well as remomves all of the outliers available in predefined
### set of columns

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

### CHECK - Function takes dataset and number of clusters as input and returns plot
### that allows to choose the correct number of clusters

def df_cleanup(dataset,columns):
    for column in columns:
        try:
            del dataset[column]
        except KeyError:
            pass
        
### Function takes a dataset, a list of columns that should be 'dummified', 
###'dummifies' this column and removes it in the end
            
def dummify_dataset(dataset, columns):
    
    import pandas as pd
    
    for column in columns:
        dataset = pd.concat([dataset, pd.get_dummies(dataset[column])],axis=1)    
        del dataset[column]
    
    return dataset

def elbow_method (df,no_clusters):

    import matplotlib.pyplot as plt

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

### Function returns heatmap based on correlation matrix and top x variables in
### terms of correlation strength with variable y

def heatmap_and_corr(dataset, column, top_correlations):
    
    import seaborn as sns
    
    corr = dataset.corr()
    
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    
    print(abs(corr[column]).sort_values(ascending = False)[1:top_correlations + 1])

### Function takes number of day and returns 0 if it's a weekday and 1 if 
### it is on weeekend

def is_weekend(row): ##needs one-hot encode
    if row['day'] >= 0 and row['day'] <= 4:
        val = 0
    else:
        val = 1
    return val

### Function takes as input the two-letter codes and assigns the geographical
### region in US. Input variables are the dataset (train, test) and pd_df_column
### which contains the encoded state variable

def map_us_region(dataset, pd_df_column):

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

    dataset['US_Region'] = dataset[pd_df_column].map(region_mapping)
    del dataset['state']

### Function takes pandas Dataframe as input and returns
### summary how many percent of all columns are NULL

def null_summary(dataset):

    for i in dataset.columns:
        print(i,round(dataset[i].isnull().sum()/len(dataset[i]),2)*100)

### Function takes pandas DataFrame as input and reduces the memory usage by 
### reducing the size of inf/float size 

def reduce_memory_usage(df, verbose = True):  
    
    import numpy as np
    
    memory_old = df.memory_usage().sum()/1024**2
    
    num = ['int16', 'int32', 'int8', 'float16', ' float32', ]
    
    for i in df.columns:
        
        if str(df[i].dtype)[:3] == 'int':
            if np.abs(df[i]).max() <= np.iinfo(np.int8).max:
                df[i] = df[i].astype(np.int8)
            elif np.abs(df[i]).max() <= np.iinfo(np.int16).max:
                df[i] = df[i].astype(np.int16)
            elif np.abs(df[i]).max() <= np.iinfo(np.int32).max:
                df[i] = df[i].astype(np.int32)
                
        if str(df[i].dtype)[:5] == 'float':
            if np.abs(df[i]).max() <= np.finfo(np.float16).max:
                df[i] = df[i].astype(np.float16)
            elif np.abs(df[i]).max() <= np.finfo(np.float32).max:
                df[i] = df[i].astype(np.float32)

    memory_new = df.memory_usage().sum()/1024**2
    if verbose == True:
        print('Memory usage before: {:5.2f} MB\nMemory usage after: {:5.2f} MB\nMemory usage decreasing: {:5.2f} %\n'.\
              format(memory_old,memory_new, (memory_old-memory_new)*100/memory_old))
    
    
    return df


### Function takes as input list of objects that are not needed at some point and
### removes them
    
def remove_objects(objects):
    for element in objects:
        try:
            exec('del ' + element)
        except:
            pass

        
### Function takes hour (integer) as input and returns time of day using rules:
### Morning [0] - from 6 to 11, Afternoon [1] - from 12 to 19, 
### Evening [2] - from 20 to 5
### Function takes optional values to define beginning and end times of morning
### and afternoon

def time_of_day(row, m_start = None, m_end = None, 
                a_start = None, a_end = None): ##!!## needs one-hot encode
    
    if m_start is None:
    
        if row['hour'] >= 6 and row['hour'] <= 11:
            val = 0
        elif row['hour'] >= 12 and row['hour'] <= 19:
            val = 1
        else:
            val = 2
        return val
    
    else:
        
        if row['hour'] >= m_start and row['hour'] <= m_end:
            val = 0
        elif row['hour'] >= a_start and row['hour'] <= a_end:
            val = 1
        else:
            val = 2
        return val
    
