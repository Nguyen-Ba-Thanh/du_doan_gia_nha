import pandas as pd 
import numpy as np

data=pd.read_csv("./data/train.csv",index_col="Id")
print(data.head())
print(data.columns)
# chon loc cac columns dac trung(features)
features= ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

x=data[features]
print(x)
y=data.iloc[:,-1]
print("y la",y)

# encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[{"encoder",OneHotEncoder(),0}])
# train/test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)

x_test=ss.transform(x_test)

# train
from sklearn.tree import DecisionTreeRegressor
data_dt=DecisionTreeRegressor()
data_dt.fit(x_train,y_train)
y_predict=data_dt.predict(x_test)
check=pd.DataFrame({"y":y_test,
       "y_predict":y_predict})
print(check)
































# # spliting
# x=data[features]
# print(x.head())
# y=data.iloc[:,-1]
# print(y.head())

# # encoding
# from sklearn.model_selection import train_test_split
# x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2,random_state=42)
# print(x_train.head())

# # training ml
# from sklearn.tree import DecisionTreeRegressor
# dt_molde=DecisionTreeRegressor(random_state=42)
# dt_molde.fit(x_train,y_train)
# y_predict=dt_molde.predict(x_valid.head())
# print(y_predict)
# check=pd.DataFrame({"y":y_valid.head(),
#                     "y_predict":y_predict})
# print(check)





