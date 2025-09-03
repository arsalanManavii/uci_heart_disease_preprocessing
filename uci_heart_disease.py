

# import libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.reset_option('display.max_columns')

dataset = pd.read_csv("heart_disease_uci.csv")
dataset

dataset.shape

dataset.isnull().sum().sort_values(ascending=False)

dataset.info()

# prepare features and lable

dataset = dataset.drop(columns=['id'])

dataset.drop(columns=['dataset'],inplace=True)

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X

y

X['fbs'] = X['fbs'].map({False:0,True:1})
X['exang'] = X['exang'].map({False:0,True:1})

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train

y_train

X_test

y_test

# handling missing values

dataset.dropna(subset=['restecg','chol'],inplace=True)

dataset.select_dtypes(include=[np.int64,np.float64]).isnull().sum().sort_values(ascending = False)

dataset.select_dtypes(include=['object']).isnull().sum().sort_values(ascending = False)

from sklearn.impute import SimpleImputer
cols_to_impute = ['ca', 'oldpeak', 'trestbps','thalch']
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

obcols_to_impute = ['thal','slope','fbs','exang']
object_imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X_train[obcols_to_impute] = object_imputer.fit_transform(X_train[obcols_to_impute])
X_test[obcols_to_impute] = object_imputer.transform(X_test[obcols_to_impute])

from sklearn.preprocessing import OrdinalEncoder
ordinalEncoder = OrdinalEncoder()
X_train[['slope','restecg']] = ordinalEncoder.fit_transform(X_train[['slope','restecg']])
X_test[['slope','restecg']] = ordinalEncoder.transform(X_test[['slope','restecg']])

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['thal','sex','cp']),('scale',StandardScaler(),['age','trestbps','chol','thalch','oldpeak'])],remainder='passthrough')
X_train_new = ct.fit_transform(X_train)
X_test_new = ct.transform(X_test)

X_train

X_train_new[0]


sns.heatmap(data=dataset.select_dtypes(include=[int,float]).corr(),annot=True,fmt='.2f')
plt.show()

dataset.select_dtypes(include=[int,float]).corr()

