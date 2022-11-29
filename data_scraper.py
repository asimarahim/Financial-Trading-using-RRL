import yfinance as yf
import pandas as pd
import datetime


# Reference yfinance tutorial here: https://towardsdatascience.com/downloading-historical-stock-prices-in-python-93f85f059c1f
# and here: https://blog.quantinsti.com/download-futures-data-yahoo-finance-library-python/
def retrieve_data(asset, start, end):
    # print stock infor to confirm I pulled the intended asset
    security_info = yf.Ticker(asset).info
    # stock_info.keys() for other properties you can explore
    print(security_info)
    
    # create empty dataframe
    security_final = pd.DataFrame()

    try:
        # download the stock price 
        security = []
        security = yf.download(asset, start=start, end=end, progress=False)
    # append the individual stock prices 
        if len(security) == 0:
            None
        else:
            security['Name']=asset
            security.index = pd.to_datetime(security.index)
            security_final = security_final.append(security,sort=False)
    except Exception:
        None
    
    return security_final
    

def main():
    # set security details and time range
    exxon = 'XOM'
    corn = 'ZC=F'
    nyse = '^NYA'
    start = datetime.datetime(2000,12,31)
    end = datetime.datetime(2021,12,4)

    # Data pull
    exxon_data = retrieve_data(exxon, start, end)
    #print(exxon_data.tail())
    exxon_data.to_csv('Exxon_20yr_OHLC.csv')

    corn_data = retrieve_data(corn, start, end)
    #print(corn_data.tail())
    exxon_data.to_csv('Corn_20yr_OHLC.csv')

    nyse_data = retrieve_data(nyse, start, end)
    #print(nyse_data.tail())
    nyse_data.to_csv('NYSE_20yr_OHLC.csv')    

if __name__ == "__main__":
    main()