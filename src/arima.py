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


train_set = train_data.iloc[:,-1].values
test_set = test_data.iloc[:, -1].values


train_set = train_set.reshape(-1, 1)
test_set = test_set.reshape(-1, 1)



train_sc = MinMaxScaler()
test_sc = MinMaxScaler()
train_set_sc = train_sc.fit_transform(train_set)
test_set_sc = test_sc.fit_transform(test_set)
min_test = test_sc.data_min_[-1]
max_test = test_sc.data_max_[-1]
print(max_test , min_test)
num_samples = train_set_sc.shape[0]
num_features = train_set_sc.shape[1]

def evaluate_model(order):

	model = ARIMA(endog=train_set_sc[:, -1], order=order)
	model_fit = model.fit()


	prediction = model_fit.predict(start=2, end=test_set.shape[0]+1)
	prediction = test_sc.inverse_transform(prediction.reshape(-1, 1))
	yhat = prediction[: , -1]

	y_true = test_set[: , -1]

	#print (yhat)
	mse = np.sqrt(mean_squared_error(y_true, yhat))
	print ("order:", str(order), ", error:", mse)

	return mse


res = coint_johansen(train_set_sc, -1 , 1).eig

best_score = 329423432424978423
best_order = 0
i = [0, 1, 2, 4, 6, 8, 10]
j = range(0, 3)
k = range(0, 3)

for p in i:
	for d in j:
		for q in k:
			order = (p,d,q)
			try:
				mse = evaluate_model(order)
				if mse < best_score:
					best_score = mse
					best_order = order
			except:
				continue
print (best_score, best_order)

"""
plt.plot(y_true, color='red', label='Actual Exchange Rate')
plt.plot(yhat, color='blue', label='Predicted Exchange Rate')
plt.title("USD/INR Exchange Rate prediction")
plt.xlabel("Time")
plt.ylabel("Exchange Rate")
#plt.ylim(64.4, 64.5)
plt.legend()
plt.show()
"""
#for i in range(len(yhat)):
#	print(abs(yhat[i] - y_true[i]))
# print(prediction)

#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/