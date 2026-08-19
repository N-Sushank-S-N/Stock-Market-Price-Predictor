import yfinance as yf
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=100)
def fetch_stock_data(ticker, years=2):
    """Fetches full historical stock data with caching."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    # We download full data, including Open, High, Low, Volume
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if df.empty:
        raise ValueError(f"Could not retrieve data for {ticker}. Please check the symbol.")
        
    # FIX: yfinance recently updated to MultiIndex columns. We must flatten them to prevent formatting crashes.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Returning full DataFrame
    return df