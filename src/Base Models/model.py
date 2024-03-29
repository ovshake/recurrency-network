import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('../data/ReCurrency-dataset-with-usa-inf-fdi.csv', index_col = 0)
data = data[:2121]

X = data.iloc[:,1:10].values
Y = data.iloc[:,11].values

scalar = StandardScaler()

X = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

# Linear regression

regressor = LinearRegression().fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print (np.sqrt(mean_squared_error(y_test, y_pred)))



########################################################################################################
#Polynomial Regression

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)
X__ = poly.fit_transform(X_test)

clf = LinearRegression()
clf.fit(X_, y_train)

y_pred = clf.predict(X__)


print (np.sqrt(mean_squared_error(y_test, y_pred)))

