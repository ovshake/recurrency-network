#Classification on Change

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../data/ReCurrency-dataset-with-usa-inf-fdi.csv', index_col = 0)
data = data[:2121]



X = data.iloc[:,1:10].values
Y = data.iloc[:,10].values

scalar = StandardScaler()

X = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

C = []
gamma = []

for i in range(1, 10, 1):
	C.append(10e-1 * i)
	C.append(10e-2 * i)
	C.append(10e-3 * i)
	C.append(10e1 * i)
	C.append(10e2 * i)
	C.append(10e3 * i)
	C.append(10e4 * i)
	C.append(10e5 * i)
	C.append(10e6 * i)
	C.append(10e7 * i)
	gamma.append(10e-1 * i)
	gamma.append(10e-2 * i)
	gamma.append(10e-3 * i)
	gamma.append(10e-4 * i)
	gamma.append(10e1 * i)
	gamma.append(10e2 * i)
	gamma.append(10e3 * i)
	gamma.append(10e4 * i)
	gamma.append(10e5 * i)
	gamma.append(10e6 * i)
	
parameters = {'C' : C, 'gamma' : gamma}


svc = SVC(kernel='rbf')

clf = GridSearchCV(svc, parameters, cv=5)

clf.fit(X_train, y_train)

print (clf.get_params())

y_pred = clf.predict(X_test)

print (np.sum(y_pred == y_test) / y_pred.shape[0])
