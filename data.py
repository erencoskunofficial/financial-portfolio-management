import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data"

def load_data(tickers, start="2020-01-01", end="2025-12-01", train_end="2025-01-01"):
 

    raw = yf.download(tickers, start=start, end=end)
    if "Adj Close" in raw.columns.get_level_values(0):
        prices = raw["Adj Close"]
    else:
        prices = raw["Close"]

    prices = prices.ffill().dropna()
    
    # verileri egitim ve test olarak ayir
    train_prices = prices[prices.index < train_end]
    test_prices = prices[prices.index >= train_end]
    
    # klasoru olustur ve csv dosyalarina kaydet
    os.makedirs(DATA_DIR, exist_ok=True)
    prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
    train_prices.to_csv(os.path.join(DATA_DIR, "train_prices.csv"))
    test_prices.to_csv(os.path.join(DATA_DIR, "test_prices.csv"))

    print(f"Kaydedilen yol: {DATA_DIR}/")
    
    return prices, train_prices, test_prices
