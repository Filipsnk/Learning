### Do wykorzystania ###

# Obliczanie ile wartosci NA jest do zastapienia
# Slownik do zastepowania NA #
# Tworzenie funkcji do zastepowania (zeby moc wdrozyc na zbiorze uczacym i testowym)
# Unskew dataset - pozbawianie skosnosci#
# np.where
# Zapis alpha do modelu
# Isfile -> sprawdzamy czy istnieje jakis plik
# Lars Lasso Coefficients
# Porownywanie Grid Search na wykresie
# Wykres Feature Importances


##### Import of the libraries #####

import time
import numpy as np,  pandas as pd, seaborn as sns, matplotlib.pyplot as plt, scipy as sp, matplotlib.patches as mpatches

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, RepeatedKFold, KFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV, 
                                  LinearRegression, LogisticRegression, Perceptron)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from sklearn.externals import joblib

#####  Setting random seed #####

RNG_SEED = int(time.time())
print("Seed: %s" % RNG_SEED)

##### Setting working directory #####

import os

#MM os.chdir('C:\\Users\Marek\\Desktop\\Python\\Kaggle\\Titanic')
os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\HousePrices')
os.chdir('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\HousePrices')

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

test = pd.read_csv('test.csv')
test_ids = import_data["Id"]
test = import_data.drop("Id", axis=1)
total_data = pd.concat((train, test)).reset_index(drop=True)

print("train set size: %s x %s" % train.shape)
print("test set size: %s x %s" % test.shape)

### Log transformation for the Sales Price variable ###

y_old = y
y = np.log1p(y)
 
plt.figure()
ax = sns.distplot(y)
ax2 = sns.distplot(y_old)

##### Available columns in the dataset #####

print("Available columns: \n")
for i in range (0,len(import_data.columns)):
    wynik = str(i) + ':' + import_data.columns[i] 
    print(wynik)

##### Assign importances to the variables #####
    
    
importances = pd.DataFrame({"Variable":import_data.columns, "Importance": np.nan})

for i in range (0,len(import_data.columns)-70):

    question = 'Jaka wartosc przypisac dla zmiennej ' + import_data.columns[i] + '?'
    importance = int(input(question))
    importances[i, 1]= importance

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

red_patch = mpatches.Patch(color='red', label='Very important')
orange_patch = mpatches.Patch(color='orange', label='Important')
yellow_patch = mpatches.Patch(color='yellow', label='Decent')
blue_patch = mpatches.Patch(color='blue', label='Medium')
navy_patch = mpatches.Patch(color='navy', label='Little important')
green_patch = mpatches.Patch(color='green', label='Unimportant')

for i in range(0,20):
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
plt.legend(handles=[red_patch, orange_patch, yellow_patch, blue_patch, navy_patch, green_patch])


##### Defining dictionary for replacement of NaNs #####

def get_replacement(data): #!# zwracamy wszystkie kolumny, ktore maja jakiekolwiek brakujace wartosci
    nb_nan_per_col = data.shape[0] - data.count() #!# info ile wartosci jest do uzupelnienia w kazdej kolumnie
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

##### Fixing the datatypes for all of the values #####

train = ordinal_object_to_str(train)
train.replace(ordinal_replacements, inplace=True)
train = fix_dtypes(train)

test = ordinal_object_to_str(test)
test.replace(ordinal_replacements, inplace=True)
test = fix_dtypes(test)

##### Unskewing all features with skewness > 0.75 #####

def unskew_dataset(data):
    numeric_features = data.dtypes[data.dtypes == float].index
    skewed_features = data[numeric_features].apply(lambda x: sp.stats.skew(x))
    skewed_features = skewed_features[skewed_features > 0.75] #!# Ustalic jaki prog zmiennych nalezy uznac za skosne i co tu jest tak naprawde liczone?
    skewed_features = skewed_features.index #!# Sprawdzic jak sie beda roznic wyniki gdy uzyjemy transformacji Box-Coxa?
    data[skewed_features] = np.log1p(data[skewed_features])
    return data

##### Feature Engineering #####

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

##### Indexes of categorical variables #####

categorical_vars_indices = np.where((train.dtypes == object))[0] #!# Dlaczego niektore zmienne maja typ 'object'?
categorical_vars = train.columns[categorical_vars_indices]

##### De-skewing of the features in train and test dataset #####

train = unskew_dataset(train)
test = unskew_dataset(test)

##### Extracting dummies #####

train_dummies = pd.get_dummies(train, columns=categorical_vars, 
                                    drop_first=True, sparse=False)
test_dummies = pd.get_dummies(test, columns=categorical_vars, 
                                 drop_first=True, sparse=False)

##### Transform dummies #####

label_enc = LabelEncoder()

for var in categorical_vars:
    var_all = pd.concat([train.loc[:, var], test.loc[:, var]])
    label_enc.fit(var_all)
    train.loc[:, var] = label_enc.transform(train.loc[:, var])
    test.loc[:, var] = label_enc.transform(test.loc[:, var])

##### Standaryzacja zmiennych ##### #!# Czy to jest tutaj niezbedne?
    
scaler = StandardScaler()

train[:] = scaler.fit_transform(train)
train_dummies[:] = scaler.fit_transform(train_dummies)
test[:] = scaler.fit_transform(test)
test_dummies[:] = scaler.fit_transform(test_dummies)

rkf_cv = KFold(n_splits=5, random_state=RNG_SEED)
stack_folds = list(KFold(n_splits=5, random_state=RNG_SEED).split(train))

##### Penalized linear regression #####

l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
alphas = alphas=[1] + [10 ** -x for x in range(1, 8)] + [5 * 10 ** -x for x in range(1, 8)]
overwrite_models = True

if not os.path.isfile("cv_opt_en.pkl") or overwrite_models:
    en_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas,
                         normalize=True, selection ="random", random_state=RNG_SEED,
                         max_iter=10000, cv=RepeatedKFold(10, 3, random_state=RNG_SEED))
    cv_opt_en = en_cv.fit(train, y)
    joblib.dump(cv_opt_en, "cv_opt_en.pkl")
else:
    cv_opt_en = joblib.load("cv_opt_en.pkl")

#####  Uzywanie cross-validated RMSE do estymacji najlepszych parametrow ##### 
l1_ratio_index = np.where(l1_ratios == cv_opt_en.l1_ratio_)[0][0]
en_alpha_index = np.where(cv_opt_en.alphas_ == cv_opt_en.alpha_)[0][0]
en_rmse = np.sqrt(np.mean(cv_opt_en.mse_path_, axis=2)[l1_ratio_index, en_alpha_index])
print(en_rmse)
print(cv_opt_en)

cv_opt_en_model = ElasticNet(alpha=cv_opt_en.alpha_, l1_ratio=cv_opt_en.l1_ratio_, 
                         fit_intercept=True, normalize=True, 
                         precompute=False, max_iter=10000, copy_X=True, tol=0.0001, 
                         warm_start=False, positive=False, random_state=RNG_SEED, 
                         selection="random")

cv_opt_en_model = cv_opt_en_model.fit(train_dummies, y)

en_preds = cv_opt_en_model.predict(test_dummies)
en_cv_preds = cross_val_predict(cv_opt_en_model, train_dummies, y, 
                                cv=stack_folds)

##### Wykres Elastic NET  #####

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(cv_opt_en.mse_path_.shape[0]):
    ax.plot(np.log10(cv_opt_en.alphas_), np.mean(cv_opt_en.mse_path_[i, :, :], axis=1),
             label=l1_ratios[i])
ax.set_title(("Elastic net regularization path (L1 / alpha vs rmse)\n"
             "best params: %s, %s" % (cv_opt_en.l1_ratio_, cv_opt_en.alpha_)))
plt.legend()

##### Elastic NET coefficients #####

fig = plt.figure(figsize=(8, 50))
ax = fig.add_subplot(111)
ax.barh(np.arange(len(cv_opt_en.coef_), 0, -1), cv_opt_en.coef_,
       tick_label=train_dummies.columns,)
ax.set_title("Elastic network coefs")
plt.show()

##### K-Nearest neighbors #####

metrics = ["euclidean", "manhattan", "minkowski", "chebyshev"]
n_neighbors_list = np.arange(4, 11, 1)

if not os.path.isfile("cv_opt_kn.pkl") or overwrite_models:
    kn = KNeighborsRegressor(n_jobs=4, p=3)
    kn_param_grid = {"n_neighbors": n_neighbors_list,
                    "weights": ["uniform", "distance"],
                    "metric": metrics}
    kn_gs = GridSearchCV(estimator=kn, param_grid=kn_param_grid, scoring="neg_mean_squared_error", 
                         fit_params=None, cv=rkf_cv)
    cv_opt_kn = kn_gs.fit(train, y)
    joblib.dump(cv_opt_kn, "cv_opt_kn.pkl")
else:
    cv_opt_kn = joblib.load("cv_opt_kn.pkl")
    
kn_rmse = np.sqrt(-cv_opt_kn.best_score_)
print(cv_opt_kn.best_score_, kn_rmse)
print(cv_opt_kn.best_estimator_)
    
cv_opt_kn_model = cv_opt_kn.best_estimator_

kn_preds = cv_opt_kn_model.predict(test)
kn_cv_preds = cross_val_predict(cv_opt_kn_model, train, y, cv=stack_folds)

uniform_run = cv_opt_kn.cv_results_["param_weights"] == "uniform"
distance_run = cv_opt_kn.cv_results_["param_weights"] == "distance"
best_metric = cv_opt_kn.best_params_["metric"]
has_best_metric = cv_opt_kn.cv_results_["param_metric"] == best_metric


##### Wykres z Grid Search #####

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(n_neighbors_list, 
        np.sqrt(-cv_opt_kn.cv_results_["mean_test_score"][uniform_run & has_best_metric]),
       label="uniform")
ax.plot(n_neighbors_list, 
        np.sqrt(-cv_opt_kn.cv_results_["mean_test_score"][distance_run & has_best_metric]),
       label="distance")
ax.set_title("Knn CV (%s) (#nn / weights vs rmse)\nBest params: %s, %s" % \
             tuple(list(cv_opt_kn.best_params_.values())))
plt.legend()
plt.show()

##### Gradient Boosting #####

if not os.path.isfile("cv_opt_xgb.pkl") or overwrite_models:
    xgb = XGBRegressor(random_state=RNG_SEED, n_estimators=500, n_jobs=4)
    reg_ratios = [0.1, 0.5, 0.9]
    xgb_param_grid = {"max_depth": [1, 2, 3, 5],
                      "learning_rate": [0.05, 0.1, 0.2],
                      "reg_lambda": reg_ratios,
                      "reg_alpha": reg_ratios}
    xgb_gs = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, 
                          scoring="neg_mean_squared_error", 
                          fit_params=None, cv=rkf_cv)
    cv_opt_xgb = xgb_gs.fit(train, y)
    joblib.dump(cv_opt_xgb, "cv_opt_xgb.pkl") 
else:
    cv_opt_xgb = joblib.load("cv_opt_xgb.pkl")
    
xgb_rmse = np.sqrt(-cv_opt_xgb.best_score_)
print(cv_opt_xgb.best_score_, xgb_rmse)
print(cv_opt_xgb.best_estimator_)

cv_opt_xgb_model = cv_opt_xgb.best_estimator_

xgb_preds = cv_opt_xgb_model.predict(test)
xgb_cv_preds = cross_val_predict(cv_opt_xgb_model, train, y, cv=stack_folds)

##### Feature importances
fig = plt.figure(figsize=(8, 30))
ax = fig.add_subplot(111)
ax.barh(np.arange(len(cv_opt_xgb_model.feature_importances_), 0, -1), 
        np.flip(np.sort(cv_opt_xgb_model.feature_importances_)),
        tick_label=train.columns)

for i in range(0,80):
    if ax.get_yticklabels()[i].properties().get('text') in very_important:
        ax.get_yticklabels()[i].set_color("red")
    elif ax.get_yticklabels()[i].properties().get('text') in important:
        ax.get_yticklabels()[i].set_color("orange")
    elif ax.get_yticklabels()[i].properties().get('text') in decent:
        ax.get_yticklabels()[i].set_color("yellow")
    elif ax.get_yticklabels()[i].properties().get('text') in medium:
        ax.get_yticklabels()[i].set_color("blue")
    elif ax.get_yticklabels()[i].properties().get('text') in little_important:
        ax.get_yticklabels()[i].set_color("navy") 
    elif ax.get_yticklabels()[i].properties().get('text') in unimportant:
        ax.get_yticklabels()[i].set_color("green") 
        
ax.set_title(cv_opt_xgb.best_score_)
plt.show()

#!# Only keep columns common to both the train and test sets

# IN [6]:

##### BRUDNOPIS #####

### 1. Petla po wszystkich kolumnach z inputboxem do wprowadzania sposobu zastepowania nulli

#wartosc = input ('Jak masz na imie?')
#print(wartosc)

### 2. Zmienna 01 czy mieszkanie sprzedane/wybudowane w okresie kryzysu w USA?    





    
