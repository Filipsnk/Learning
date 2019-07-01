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

missing_val_replace = {}
    
    # Type of zone (residential, commercial etc.), cannot be guessed 
    # with current data. Set the mode.
    missing_val_replace["MSZoning"] = total_data["MSZoning"].mode()[0]
    missing_val_replace["Utilities"] = total_data["Utilities"].mode()[0]
    test.loc[np.any(pd.isnull(test), axis=1), nb_nan_per_col != 0]
    missing_val_replace["LotFrontage"] = 0
    missing_val_replace["Alley"] = "None"
    missing_val_replace["Exterior1st"] = total_data["Exterior1st"].mode()[0]
    missing_val_replace["Exterior2nd"] = total_data["Exterior2nd"].mode()[0]
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

##### Fitting simple linear regression #####

    
