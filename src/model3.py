# Linear regression

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score 
from sklearn.svm import SVC

# data = pd.read_csv('../data/ReCurrency-dataset-with-usa-inf-fdi.csv', index_col = 0)
# data = data[:2121]



# X = data.iloc[:,1:10].values
# Y = data.iloc[:,11].values

# scalar = StandardScaler()

# X = scalar.fit_transform(X)

data = pd.read_csv('../data/ReCurrency-dataset-with-usa-inf-fdi.csv', index_col = 0)
data = data[:2121]



X = data.iloc[:,1:10].values
Y = data.iloc[:,10].values

scalar = StandardScaler()

X = scalar.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
gamma = [0.00001 , 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
C = [0.00001 , 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for c in C:
	for g in gamma:
		svm = SVC(kernel = 'rbf', C = c, gamma = g) 
		svm.fit(X_train, y_train) 
		y_pred = svm.predict(X_test) 
		score = accuracy_score(y_test , y_pred) 
		print("For C = {} and Gamma = {} Accuracy is {}".format(c , g , score)) 
