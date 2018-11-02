#This script changes the resolution for the USA inflation to make it daily in coherence with INR/USD conversion rate. 
import pandas as pd 
usa_inflation = pd.read_csv('../data/USA_Inflation_Monthly.csv')
usd_inr = pd.read_csv('../data/USD_INR Historical Data.csv')

usd_inr['Date'] = pd.to_datetime(usd_inr.Date) 
month_usa_inflation = {} 
for i in range(len(usa_inflation)):
    k = usa_inflation.loc[i , 'TIME']
    v = usa_inflation.loc[i , 'Value']
    month_usa_inflation[str(k)] = v 


usa_inf = [] 
for i in range(len(usd_inr)):
    date = usd_inr.loc[i , 'Date']
    date = str(date).split("-") 
    date = date[0] + "-" + date[1] 
    try:
        usa_inf.append(month_usa_inflation[str(date)]) 
    except:
        usa_inf.append(-1) 


usd_inr['USA Inflation'] = usa_inf 
# print(usa_inflation.head())
print(usd_inr.head(100)) 

usd_inr.to_csv('../data/USD INR Combined with USA Inflation rate.csv')

