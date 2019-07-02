### Do wykorzystania ###

# Slownik do zastepowania NA #
# Tworzenie funkcji do zastepowania (zeby moc wdrozyc na zbiorze uczacym i testowym)
# Unskew dataset - pozbawianie skosnosci#
# np.where

##### Import of the libraries #####
import time
import numpy as np,  pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import scipy as sp
##### Setting working directory #####

import os

##### 
RNG_SEED = int(time.time())
print("Seed: %s" % RNG_SEED)

#MM os.chdir('C:\\Users\Marek\\Desktop\\Python\\Kaggle\\Titanic')
os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\HousePrices')

##### Related kernel #####

import webbrowser
webbrowser.open_new_tab('https://www.kaggle.com/msimon/house-prices-model-stacking')

##### Ignore warnings ##### 

import warnings
warnings.filterwarnings('ignore')

##### Import & Concat of the data ##### 
##### Reorder according to predicted usability #####

import_data = pd.read_csv('train.csv')

train_ids = import_data["Id"]
train = import_data.drop(["Id", "SalePrice"], axis=1)

X = import_data.loc[:, import_data.columns != 'SalePrice']
y = import_data.loc[:, import_data.columns == 'SalePrice']

### Log transformation for the Sales Price variable ###
y = np.log1p(y)

test = pd.read_csv('test.csv')
test_ids = import_data["Id"]
test = import_data.drop("Id", axis=1)

total_data = pd.concat((train, test)).reset_index(drop=True)

print("train set size: %s x %s" % train.shape)
print("test set size: %s x %s" % test.shape)

plt.figure()
ax = sns.distplot(y)

def get_replacement(data):
# def get_replacement(train_set, test_set):
    # data = pd.concat([train_set, test_set])
    nb_nan_per_col = data.shape[0] - data.count()
    print(nb_nan_per_col[nb_nan_per_col != 0])

##### Available columns in the dataset #####

print("Available columns: \n")
for i in range (0,len(dane.columns)):
    wynik = str(i) + ':' + dane.columns[i] 
    print(wynik)

##### Predicted candidates for the regression ##### 

very_important = ['MSSubClass', 'LotArea', 'LandSlope', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'MoSold', 'YrSold']
important = ['LandContour', 'Utilities', 'OverallQual','OverallCond','Foundation','Heating', 'CentralAir','Functional', 'Fence']
decent = ['MSZoning', 'Street', 'Alley', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'BsmtQual','BsmtFinType1', 'TotalBsmtSF', 'HeatingQC', 'Electrical','BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'GarageType', 'GarageYrBlt', 'GarageArea']
medium = ['SaleType', 'LotFrontage', 'LotShape', 'BldgType', 'HouseStyle', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Fireplaces', 'FireplaceQu', 'GarageFinish', 'GarageCars', 'GarageQual','GarageCond','PavedDrive','MiscFeature', 'SaleCondition']
little_important = ['MSSubClass', 'LotConfig','Neighborhood', 'Condition1', 'Condition2', 'MasVnrType', 'MasVnrArea','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'MiscVal']
unimportant =  ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC']


##### Storing the missing data in new variables ##### 

all_data_na = (total_data.isnull().sum() / len(total_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

##### Barplot with percentage of missing data #####

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)

###https://stackoverflow.com/questions/11244514/modify-tick-label-text

for i in range(0,30):
    if ax.get_xticklabels()[i].properties().get('text') in very_important:
        ax.get_xticklabels()[i].set_color("red")
    elif ax.get_xticklabels()[i].properties().get('text') in important:
        ax.get_xticklabels()[i].set_color("orange")
    elif ax.get_xticklabels()[i].properties().get('text') in decent:
        ax.get_xticklabels()[i].set_color("yellow")
    elif ax.get_xticklabels()[i].properties().get('text') in medium:
        ax.get_xticklabels()[i].set_color("blue")
    elif ax.get_xticklabels()[i].properties().get('text') in little_important:
        ax.get_xticklabels()[i].set_color("navy") 
    elif ax.get_xticklabels()[i].properties().get('text') in unimportant:
        ax.get_xticklabels()[i].set_color("green") 

plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

##### Adjusted R2 to avoid situation, where only adding new variable increases the R2 #####

def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)



##### Defining dictionary for replacement of NaNs #####

def get_replacement(data):
    nb_nan_per_col = data.shape[0] - data.count()
    print(nb_nan_per_col[nb_nan_per_col != 0])

    missing_val_replace = {}
    missing_val_replace["MSZoning"] = total_data["MSZoning"].mode()[0]
    missing_val_replace["Utilities"] = total_data["Utilities"].mode()[0]
    missing_val_replace["LotFrontage"] = 0
    missing_val_replace["Alley"] = "None"
    missing_val_replace["Exterior1st"] = total_data["Exterior1st"].mode()[0]
    missing_val_replace["Exterior2nd"] = total_data["Exterior2nd"].mode()[0]
    missing_val_replace["MasVnrType"] = "None"
    missing_val_replace["MasVnrArea"] = 0
    missing_val_replace["BsmtFinType1"] = "Unf"
    missing_val_replace["BsmtFinType2"] = "Unf"
    missing_val_replace["BsmtQual"] = "TA"
    missing_val_replace["BsmtCond"] = "TA"
    missing_val_replace["BsmtExposure"] = total_data["BsmtExposure"].mode()[0]
    missing_val_replace["BsmtFinSF1"] = 0
    missing_val_replace["BsmtFinSF2"] = 0
    missing_val_replace["BsmtUnfSF"] = 0
    missing_val_replace["TotalBsmtSF"] = 0
    missing_val_replace["BsmttotalBath"] = 0
    missing_val_replace["BsmtHalfBath"] = 0
    missing_val_replace["Electrical"] = total_data["Electrical"].mode()[0]
    missing_val_replace["KitchenQual"] = total_data["KitchenQual"].mode()[0]
    missing_val_replace["Functional"] = total_data["Functional"].mode()[0]
    missing_val_replace["FireplaceQu"] = "None"
    missing_val_replace["GarageType"] = "None"
    missing_val_replace["GarageYrBlt"]= np.round(total_data["GarageYrBlt"].median())
    missing_val_replace["GarageCars"]= np.round(total_data["GarageCars"].median())
    missing_val_replace["GarageArea"]= np.round(total_data["GarageArea"].median())
    missing_val_replace["GarageFinish"]= "None"
    missing_val_replace["GarageQual"]= "None"
    missing_val_replace["GarageCond"] = "None"
    missing_val_replace["PoolQC"] = "None"
    missing_val_replace["Fence"] = "None"
    missing_val_replace["MiscFeature"] = "None"
    missing_val_replace["SaleType"] = total_data["SaleType"].mode()[0]
    
    return missing_val_replace

train.fillna(get_replacement(train), inplace=True)
test.fillna(get_replacement(test), inplace=True)  

print("Remaining missing values in train and test sets:")
print(np.sum((train.shape[0] - train.count()) != 0))
print(np.sum((test.shape[0] - test.count()) != 0))

##### Changing the datatype of variables #####

def ordinal_object_to_str(df):
    df["LotShape"] = df["LotShape"].astype(str)
    df["Utilities"] = df["Utilities"].astype(str)
    df["LandSlope"] = df["LandSlope"].astype(str)
    df["ExterQual"] = df["ExterQual"].astype(str)
    df["ExterCond"] = df["ExterCond"].astype(str)
    df["BsmtQual"] = df["BsmtQual"].astype(str)
    df["BsmtCond"] = df["BsmtCond"].astype(str)
    df["BsmtExposure"] = df["BsmtExposure"].astype(str)
    df["BsmtFinType1"] = df["BsmtFinType1"].astype(str)
    df["BsmtFinType2"] = df["BsmtFinType2"].astype(str)
    df["HeatingQC"] = df["HeatingQC"].astype(str)
    df["Electrical"] = df["Electrical"].astype(str)
    df["KitchenQual"] = df["KitchenQual"].astype(str)
    df["Functional"] = df["Functional"].astype(str)
    df["FireplaceQu"] = df["FireplaceQu"].astype(str)
    df["GarageQual"] = df["GarageQual"].astype(str)
    df["GarageCond"] = df["GarageCond"].astype(str)
    df["PavedDrive"] = df["PavedDrive"].astype(str)
    df["PoolQC"] = df["PoolQC"].astype(str)
    df["Fence"] = df["Fence"].astype(str)
    return df

def fix_dtypes(df):

    df["MSSubClass"] = df["MSSubClass"].astype(object)
    
    df["LotShape"] = df["LotShape"].astype(int)
    df["Utilities"] = df["Utilities"].astype(int)
    df["LandSlope"] = df["LandSlope"].astype(int)
    df["ExterQual"] = df["ExterQual"].astype(int)
    df["ExterCond"] = df["ExterCond"].astype(int)
    df["BsmtQual"] = df["BsmtQual"].astype(int)
    df["BsmtCond"] = df["BsmtCond"].astype(int)
    df["BsmtExposure"] = df["BsmtExposure"].astype(int)
    df["BsmtFinType1"] = df["BsmtFinType1"].astype(int)
    df["BsmtFinType2"] = df["BsmtFinType2"].astype(int)
    df["HeatingQC"] = df["HeatingQC"].astype(int)
    df["Electrical"] = df["Electrical"].astype(int)
    df["KitchenQual"] = df["KitchenQual"].astype(int)
    df["Functional"] = df["Functional"].astype(int)
    df["FireplaceQu"] = df["FireplaceQu"].astype(int)
    df["GarageQual"] = df["GarageQual"].astype(int)
    df["GarageCond"] = df["GarageCond"].astype(int)
    df["PavedDrive"] = df["PavedDrive"].astype(int)
    df["PoolQC"] = df["PoolQC"].astype(int)
    df["Fence"] = df["Fence"].astype(int)
    df["GarageYrBlt"] = df["GarageYrBlt"].astype(int)

    df["LotArea"] = df["LotArea"].astype(float)
    df["BsmtFinSF1"] = df["BsmtFinSF1"].astype(float)
    df["BsmtFinSF2"] = df["BsmtFinSF2"].astype(float)
    df["BsmtUnfSF"] = df["BsmtUnfSF"].astype(float)
    df["TotalBsmtSF"] = df["TotalBsmtSF"].astype(float)
    df["1stFlrSF"] = df["1stFlrSF"].astype(float)
    df["2ndFlrSF"] = df["2ndFlrSF"].astype(float)
    df["LowQualFinSF"] = df["LowQualFinSF"].astype(float)
    df["GrLivArea"] = df["GrLivArea"].astype(float)
    df["GarageArea"] = df["GarageArea"].astype(float)
    df["WoodDeckSF"] = df["WoodDeckSF"].astype(float)
    df["OpenPorchSF"] = df["OpenPorchSF"].astype(float)
    df["EnclosedPorch"] = df["EnclosedPorch"].astype(float)
    df["3SsnPorch"] = df["3SsnPorch"].astype(float)
    df["ScreenPorch"] = df["ScreenPorch"].astype(float)
    df["PoolArea"] = df["PoolArea"].astype(float)
    df["MiscVal"] = df["MiscVal"].astype(float)
    
    return df

ordinal_replacements = {}
ordinal_replacements["LotShape"] = {"Reg": "0", "IR1": "1", "IR2": "2", "IR3": "3"}
ordinal_replacements["Utilities"] = {"AllPub": "0", "NoSewr": "1", "NoSeWa": "2", "ELO": "3"}
ordinal_replacements["LandSlope"] = {"Gtl": "0", "Mod": "1", "Sev": "2"}
ordinal_replacements["ExterQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["ExterCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["BsmtQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["BsmtCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["BsmtExposure"] = {"Gd": "0", "Av": "1", "Mn": "2", "No": "3"}
ordinal_replacements["BsmtFinType1"] = {"GLQ": "0", "ALQ": "1", "BLQ": "2", "Rec": "3", "LwQ": "4", "Unf": "5"}
ordinal_replacements["BsmtFinType2"] = {"GLQ": "0", "ALQ": "1", "BLQ": "2", "Rec": "3", "LwQ": "4", "Unf": "5"}
ordinal_replacements["HeatingQC"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["Electrical"] = {"SBrkr": "0", "FuseA": "1", "FuseF": "2", "FuseP": "3", "Mix": "4"}
ordinal_replacements["KitchenQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4"}
ordinal_replacements["Functional"] = {"Typ": "0", "Min1": "1", "Min2": "2", "Mod": "3", "Maj1": "4", 
                                      "Maj2": "5", "Sev": "6", "Sal": "7"}
ordinal_replacements["FireplaceQu"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}
ordinal_replacements["GarageQual"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}
ordinal_replacements["GarageCond"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}
ordinal_replacements["PavedDrive"] = {"Y": "0", "P": "1", "N": "2"}
ordinal_replacements["PoolQC"] = {"Ex": "0", "Gd": "1", "TA": "2", "Fa": "3", "Po": "4", "None": 5}
ordinal_replacements["Fence"] = {"GdPrv": "0", "MnPrv": "1", "GdWo": "2", "MnWw": "3", "None": 5}


train = ordinal_object_to_str(train)
train.replace(ordinal_replacements, inplace=True)
train = fix_dtypes(train)

test = ordinal_object_to_str(test)
test.replace(ordinal_replacements, inplace=True)
test = fix_dtypes(test)


def unskew_dataset(data):
    numeric_features = data.dtypes[data.dtypes == float].index
    skewed_features = data[numeric_features].apply(lambda x: sp.stats.skew(x)) #compute skewness
    skewed_features = skewed_features[skewed_features > 0.75]
    skewed_features = skewed_features.index
    data[skewed_features] = np.log1p(data[skewed_features])
    return data

##### Feature Engineering #####

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

##### Indexes of categorical variables #####

categorical_vars_indices = np.where((train.dtypes == object))[0]
categorical_vars = train.columns[categorical_vars_indices]

##### De-skewing of the features in train and test dataset #####

train = unskew_dataset(train)
test = unskew_dataset(test)


##### Extracting dummies #####

train_dummies = pd.get_dummies(train, columns=categorical_vars, 
                                    drop_first=True, sparse=False)
test_dummies = pd.get_dummies(test, columns=categorical_vars, 
                                 drop_first=True, sparse=False)


#!# Only keep columns common to both the train and test sets

# IN [6]:

##### BRUDNOPIS #####

### 1. Petla po wszystkich kolumnach z inputboxem do wprowadzania sposobu zastepowania nulli

#wartosc = input ('Jak masz na imie?')
#print(wartosc)

### 2. Zmienna 01 czy mieszkanie sprzedane/wybudowane w okresie kryzysu w USA?    





    
