import pandas as pd 
import numpy as np 
# from sklearn.preprocessing import StandardScaler 

data = pd.read_csv('../data/ReCurrency-dataset-with-usa-inf-fdi.csv', index_col = 0)
data = data[:2122]
# Linear regression, SVRs, polynomial regression 
data['Prev Day Price'] = data['Price'].shift(1) 
data['Change'] = (data['Price'] - data['Prev Day Price']) >= 0
for i in range(len(data)):
    if data.loc[i , 'Change']:
        data.loc[i , 'Change'] = 1
    else:
        data.loc[i, 'Change'] = -1


print(data.tail())

