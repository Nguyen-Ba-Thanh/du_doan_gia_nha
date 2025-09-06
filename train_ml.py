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






































