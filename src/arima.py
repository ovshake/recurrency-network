import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import warnings
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
warnings.filterwarnings("ignore")

train_data = pd.read_csv('../data/train_data.csv' , index_col = 0)
test_data = pd.read_csv('../data/test_data.csv' ,index_col = 0)
train_set = train_data.iloc[:,1:].values
test_set = test_data.iloc[: , 1:].values
# print(test_set)
# print(train_set.shape)
sc = MinMaxScaler()
train_set_sc = sc.fit_transform(train_set)
test_set_sc = sc.fit_transform(test_set)
num_samples = train_set_sc.shape[0]
num_features = train_set_sc.shape[1]
# X_train = train_set_sc[:, :6]
# X_train = X_train[:-1] 
# Y_train =  train_set_sc[: , -1]
# Y_train = Y_train[1:]
# X_test = test_set_sc[:, :6]
# X_test = X_test[:-1] 
# Y_test =  test_set_sc[: , -1]
# Y_test = Y_test[1:]
# print(Y_train)
# adf = sm.tsa.stattools.adfuller(Y_train,autolag = 't-stat')
# print(adf)


res = coint_johansen(train_set_sc, -1 , 1).eig
# print(res)

model = VAR(endog=train_set_sc)
model_fit = model.fit()
prediction = model_fit.forecast(model_fit.y, steps=len(test_set_sc))
yhat = prediction[: , -1]
print(yhat)
y_true = test_set_sc[: , -1]
print(y_true)
for i in range(len(yhat)):
	print(abs(yhat[i] - y_true[i]))
# print(prediction)