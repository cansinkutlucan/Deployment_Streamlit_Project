import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.display.float_format = '{:.3f}'.format 


#%% EDA
df = pd.read_csv('final_scout_not_get_dummy.csv')
df.columns = [x.lower() for x in df.columns]
df_ = df.copy()

df.head()
df.info()
df.describe()
df.isnull().sum()
df.corr()

for model in df.make_model.unique():
    car_prices = df[df["make_model"]== model]["price"]
    Q1 = car_prices.quantile(0.25)
    Q3 = car_prices.quantile(0.75)
    IQR = Q3-Q1 
    lower_lim = Q1-1.5*IQR
    upper_lim = Q3+1.5*IQR
    drop_index = df[df["make_model"]== model][(car_prices < lower_lim) | (car_prices > upper_lim)].index
    df.drop(index = drop_index, inplace=True)

df_object = df.select_dtypes(include ="object").head()
df_object

for col in df_object:
    print(f"{col:<20}:", df[col].nunique())

ax = df.make_model.value_counts().plot(kind ="bar")
ax.bar_label(ax.containers[0]);

sns.histplot(df.price, bins=50, kde=True)
sns.boxplot(df.price)
sns.heatmap(df.corr(), annot =True)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

df_num = df.select_dtypes(include='number')

for col in df_num:
    print(col, check_outlier(df, col))  # km-previous_owners-hp_kw-co2 emission # 4

# Let's define evaluation function here. We will use it below in model evaluation
def train_val(model, X_train, y_train, X_test, y_test):    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)   
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},   
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}   
    return pd.DataFrame(scores)

#%% MODELLING
import xgboost as xgb
# xgb.__version__ # 0.90
# #!pip install xgboost==0.90
from xgboost import XGBRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
# Train_Test_Split
X = df_.drop(columns="price")
y = df_.price

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Modeling with Pipeline
cat = X.select_dtypes("object").columns
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')

operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, objective="reg:squarederror"))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

train_val(pipe_model, X_train, y_train, X_test, y_test)

# Cross validation
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, objective="reg:squarederror"))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores).iloc[:, 2:].mean() # 0.938

# Grid Search
param_grid = {"XGB_model__n_estimators":[300,500,700],"XGB_model__max_depth":[2,4,6], "XGB_model__learning_rate": [0.08,0.1,0.15,0.2],"XGB_model__subsample":[0.3,0.5,0.7], "XGB_model__colsample_bytree":[0.5, 1]}
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(objective="reg:squarederror"))]
model = Pipeline(steps=operations)
grid_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error',cv=5,n_jobs = -1)
grid_model.fit(X_train,y_train)
grid_model.best_params_
grid_model.best_estimator_
grid_model.best_score_

train_val(grid_model, X_train, y_train, X_test, y_test)

# Cross Validation
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(learning_rate=0.08, max_depth=6, n_estimators=700, objective='reg:squarederror',subsample=0.7,colsample_bytree=0.5))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)
pd.DataFrame(scores).mean()[2:]  # 0.948

# Feature Importance for XGBoost
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(learning_rate=0.05, max_depth=2, n_estimators=300, objective='reg:squarederror', random_state=101,subsample=0.5))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X_train, y_train)

features = list(X.select_dtypes("object").columns) + list(X.select_dtypes("number").columns) 
features

pipe_model["XGB_model"].feature_importances_

df_f_i = pd.DataFrame(data = pipe_model["XGB_model"].feature_importances_, index=features, columns = ["Feature Importance"]).sort_values("Feature Importance", ascending=False)
df_f_i  # hp_kw , gears, age, make_model , km

def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model testing performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")

#%% DEPLOYMENT
from xgboost import XGBRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

new_list = ["age", "hp_kw", "km", "gears", "make_model"]
X = df[new_list]
y = df['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Modeling with Pipeline
cat = X.select_dtypes("object").columns
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')

operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101, objective="reg:squarederror"))]
xgb_model = Pipeline(steps=operations)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

eval_metric(y_test, y_pred)


import pickle
pickle.dump(xgb_model, open('xgb_model_new1','wb')) 

# X.describe().T
X.make_model.nunique()
"""
X = pd.get_dummies(X)
X.sample(5)   # all numeric
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

xgb_model = XGBRegressor(learning_rate=0.08, max_depth=6, n_estimators=700, objective='reg:squarederror',subsample=0.7,colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
eval_metric(y_test, y_pred) # R2 : 0.92 , rmse: 1785.75
import pickle
pickle.dump(xgb_model, open('xgb_model_new','wb')) 
"""




