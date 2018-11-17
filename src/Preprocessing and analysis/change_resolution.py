#This script changes the resolution for the USA inflation to make it daily in coherence with INR/USD conversion rate. 
import pandas as pd 
def append_USA_inflation_rate():
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

# def append_to_main():


main_dataset = pd.read_csv('../data/ReCurrency-dataset.csv')
main_dataset['Date'] = pd.to_datetime(main_dataset['Date'])
USA_inflation = pd.read_csv('../data/USD INR Combined with USA Inflation rate.csv' )
USA_inflation['Date'] = pd.to_datetime(USA_inflation['Date'])
USA_inflation = USA_inflation[['Date' , 'USA Inflation']]
main_dataset = pd.merge(main_dataset , USA_inflation, on='Date', how = 'left')

main_dataset.to_csv('../data/ReCurrency-dataset-with-usa-inf.csv')
