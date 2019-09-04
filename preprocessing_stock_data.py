import bs4 as bs
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import requests


def get_companies_data():
    
    # getting companies names and put those in names_list
    re = requests.get('https://www.slickcharts.com/sp500')
    bsp = bs.BeautifulSoup(re.text, 'lxml')
    target = bsp.find('table', {'class': 'table table-hover table-borderless table-sm'})
    names_list = []
    for i in target.findAll('tr')[1:]:
        name = i.findAll('td')[2].text
        names_list.append(name.rstrip())
    

    start_date = dt.datetime(2001, 1, 1)
    today = dt.datetime.now()
    
    # getting actual data from yahoo and store those
    for i in names_list:
        try:
            df = web.DataReader(i.replace('.','-'),'yahoo',start_date,today)
            df.to_csv('Kabu/'+i+'.csv')
        except:
            print('Cannot get' +i)
    return names_list

def extract_close_data(names_list):
    
    # extract only adjust close data from each files
    close_df = pd.DataFrame()
    for i in names_list:
        df = pd.read_csv('Kabu/'+i+'.csv')
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': i}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        close_df = close_df.join(df, how='outer')

    close_df.to_csv('stock_asjclosed.csv')

def main():
    names_list = get_companies_data()
    extract_close_data(names_list)
    
if __name__ == '__main__':
    main()
    