import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import warnings
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

train_data = pd.read_csv('../data/train_data.csv' , index_col = 0)
test_data = pd.read_csv('../data/test_data.csv' ,index_col = 0)
train_set = train_data.iloc[:,1:].values
test_set = test_data.iloc[: , 1:].values



# print(test_set)
# print(train_set.shape)

train_sc = MinMaxScaler()
test_sc = MinMaxScaler()
train_set_sc = train_sc.fit_transform(train_set)
test_set_sc = test_sc.fit_transform(test_set)
min_test = test_sc.data_min_[-1]
max_test = test_sc.data_max_[-1]
print(max_test , min_test)
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
prediction = test_sc.inverse_transform(prediction)
yhat = prediction[: , -1]
# print(yhat)
y_true = test_set[: , -1]
# print(y_true)
print (yhat)

print (np.sqrt(mean_squared_error(y_true, yhat)))

plt.plot(y_true, color='red', label='Actual Exchange Rate')
plt.plot(yhat, color='blue', label='Predicted Exchange Rate')
plt.title("USD/INR Exchange Rate prediction")
plt.xlabel("Time")
plt.ylabel("Exchange Rate")
#plt.ylim(64.4, 64.5)
plt.legend()
plt.show()

#for i in range(len(yhat)):
#	print(abs(yhat[i] - y_true[i]))
# print(prediction)

#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/