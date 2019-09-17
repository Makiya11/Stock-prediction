# Stock-prediction

This program will predict if we should buy, sell, hold stock.

It extracted 500 companies past stock data(18 years) in csv file using Web scraping

After that putiting labels for the company's data a user wants to predict

This is how label was created:
if stock data go up more than XX% in 5 days,buy(1)
if stock data decrease more than XX% in 5 days, sell(-1)
if stock data did change more than XX% in 5 days, hold(0)

Then machine learning clasiffer will learn based on this imformation and predict buy, sell, hold. 
