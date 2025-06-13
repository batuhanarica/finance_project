import yfinance as yf
import pandas as pd
import numpy as np
import data_loader

def compute_daily_simple_returns(df):
    """
    Compute daily returns for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with stock prices.

    Returns:
    pd.DataFrame: DataFrame with daily returns.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column to compute daily returns.")
    
    df['Daily Return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())
    return df

def compute_daily_log_returns(df):
    """
    Compute daily log returns for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with stock prices (must include 'Close' and 'Ticker').

    Returns:
    pd.DataFrame: DataFrame with a new column 'Log Return' for each ticker.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column to compute log returns.")

    df['Log Return'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    return df

def compute_cumulative_log_returns(df):
    """
    Compute cumulative log returns for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with stock prices (must include 'Close' and 'Ticker').

    Returns:
    pd.DataFrame: DataFrame with a new column 'Cumulative Log Return' for each ticker.
    """
    if 'Log Return' not in df.columns:
        raise ValueError("DataFrame must contain 'Log Return' column to compute cumulative log returns.")

    df['Cumulative Log Return'] = df.groupby('Ticker')['Log Return'].transform(lambda x: x.cumsum())
    return df

def cumulative_simple_daily_returns(df):
    """
    Compute cumulative simple returns for the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with stock prices (must include 'Close' and 'Ticker').

    Returns:
    pd.DataFrame: DataFrame with a new column 'Cumulative Simple Return' for each ticker.
    """
    if 'Daily Return' not in df.columns:
        raise ValueError("DataFrame must contain 'Daily Return' column to compute cumulative simple returns.")

    df['Cumulative Simple Return'] = df.groupby('Ticker')['Daily Return'].transform(lambda x: (1 + x).cumprod() - 1)
    return df

def compute_returns(df, method='log'):
    """
    Compute daily and cumulative returns using specified method.

    Parameters:
    df (pd.DataFrame): DataFrame with stock prices.
    method (str): 'log' or 'simple'

    Returns:
    pd.DataFrame: DataFrame with return columns added.
    """
    if method == 'log':
        df = compute_daily_log_returns(df)
        df = compute_cumulative_log_returns(df)
    elif method == 'simple':
        df = compute_daily_simple_returns(df)
        df = cumulative_simple_daily_returns(df)
    else:
        raise ValueError("method must be 'log' or 'simple'")
    
    return df

