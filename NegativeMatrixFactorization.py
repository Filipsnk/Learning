
"""
Created on Wed Mar 11 09:32:47 2020

"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sys import getsizeof
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
import time

#Import of data
names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('u.data', sep='\t', names=names)

#Pivot table not used    
data = pd.pivot_table(ratings_df[['user_id','item_id','rating']],values='rating',index='user_id',columns='item_id')
data.fillna(0,inplace = True)


'''
Function to delete columns when meet threshold

def delete_columns(df,ratio):
    
    count=0
    
    for col in df.columns:
        result = df.loc[:,col].value_counts()[0]
        
        if round(result/df.shape[0],2) > ratio:
            df = df.drop(columns=col)
            count += 1
    print('{} columns were deleted'.format(count))

    return  df
'''

## Get the shape of final matrix
n_users = len(ratings_df['user_id'].unique())
n_items = len(ratings_df['item_id'].unique())
matrix_shape = (n_users,n_items)

X = ratings_df[['user_id','item_id']].values
y = ratings_df['rating'].values


#Convert matrix to more user friendly

def ConvertMatrix(X,y,R_shape):
    rows = X[:,0]  ##users
    columns = X[:,1]##items
    data_2 = y
    
    matrix = sparse.csr_matrix((data_2,(rows,columns)),shape = (R_shape[0]+1,R_shape[1]+1))
    
    R = matrix.todense()
    R = R[1:,1:]
    R = np.asarray(R)
    
    return R


R = ConvertMatrix(X,y,matrix_shape)

'''
    Creating inital model NMF
    
    X_pred = W * H 
    

'''

components = 10

nmf_model = NMF(n_components = components)
nmf_model.fit(R)

W = nmf_model.transform(R)
H = nmf_model.components_

Pred = W.dot(H)


def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()     
    actual = actual[actual.nonzero()].flatten() 
    return mean_squared_error(pred, actual,squared = False)


#Lets check how our model performed
    
print('RMSE for NMF model with {} components equals to {}'.format(components,rmse(Pred,R)))


### Split into training and test set for evaluation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


R_train = ConvertMatrix(X_train, y_train, matrix_shape)
R_test = ConvertMatrix(X_test, y_test, matrix_shape)


parametersNMF = {
                    'n_components' : 4,     #  default number of n_components
                    'init' : 'random', 
                    'random_state' : 0, 
                    'alpha' : 0.01,          # regularization term
                    'l1_ratio' : 0,          # set regularization = L2 
                    'max_iter' : 15
                }

estimator = NMF(**parametersNMF)

err = 0
n_iter = 0
n_folds = 5

model_2 = KFold(n_splits = 5, shuffle = False)


for train_index, test_index in model_2.split(X):   
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    # Converting sparse array to dense array
    R_train = ConvertMatrix(X_train, y_train, matrix_shape)
    R_test = ConvertMatrix(X_test, y_test, matrix_shape)
    
    # Training (matrix factorization)
    t0 = time.time()
    estimator.fit(R_train)  
    Theta = estimator.transform(R_train)       # user features
    M = estimator.components_.T                # item features
    print ("Fit in %0.3fs" % (time.time() - t0))
    n_iter += estimator.n_iter_ 

    # Making the predictions
    R_pred = M.dot(Theta.T)
    R_pred = R_pred.T      
    
    # Computing the error on the validation set 
    err += get_rmse(R_pred, R_test)
    print (get_rmse(R_pred, R_test))
    
print ("*** RMSE Error : ", err / n_folds)
print ("Mean number of iterations:", n_iter / n_folds)




#### Testing but using grid search


cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0) 

param =        {
                    'n_components' : [15, 20, 25],
                    'alpha' : [0.001, 0.01, 0.1],
                    'l1_ratio' : [0], 
                    'max_iter' : [15, 20, 25]
                }

# Keep track of RMSE and parameters
grid_search = pd.DataFrame([[0, 0, 0, 0, 0]])
grid_search.columns = ['n_components', 'alpha', 'l1_ratio', 'max_iter'] + ['RMSE']

# nb of folds in ShuffleSplit CV
n_folds = 5      
i = 0

# Performing the Grid search
for n_components in param['n_components']:
    for alpha in param['alpha']:
        for l1_ratio in param['l1_ratio']:
            for max_iter in param['max_iter']:

                err = 0
                n_iter = 0
            
                
                for train_index, test_index in cv.split(X_train):
    
                    
                    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    
                    # Converting sparse array to dense array
                    R_train = ConvertMatrix(X_train_cv, y_train_cv, matrix_shape)
                    R_test = ConvertMatrix(X_test_cv, y_test_cv, matrix_shape)

                    # updating the parameters
                    parametersNMF = {
                    'n_components' : n_components,
                    'init' : 'random', 
                    'random_state' : 0, 
                    'alpha' : alpha,
                    'l1_ratio' : l1_ratio,
                    'max_iter' : max_iter}
                    
                    estimator = NMF(**parametersNMF)
                
                    # Training (matrix factorization)
                    t0 = time.time()
                    estimator.fit(R_train)  
                    Theta = estimator.transform(R_train)       # user features
                    M = estimator.components_.T                # item features
                    #print "Fit in %0.3fs" % (time.time() - t0)
                    n_iter += estimator.n_iter_ 

                    # Making the predictions
                    R_pred = M.dot(Theta.T).T
                    
                    # Computing the error on the validation set 
                    err += rmse(R_pred, R_test)
    
                #print "RMSE Error : ", err / n_folds
                grid_search.loc[i] = [n_components, alpha, l1_ratio, max_iter, err / n_folds]
                print (grid_search.loc[i].tolist(), "Mean number of iterations:", n_iter / n_folds)
                i += 1

best_params = grid_search.sort_values('RMSE')[:1]
print ('*** best params ***')
print (best_params)




















































