import yfinance as yf
import pandas as pd


#yf.enable_debug_mode()

def load_data_yf_single(ticker, start_date, end_date):
    """
    Load historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: DataFrame containing the stock data.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def load_data_yf_long_format(tickers, start_date, end_date):
    """
    Load and return long-format historical data for multiple tickers.

    Returns:
    pd.DataFrame with columns: ['Date', 'Ticker', 'Open', 'High', ...]
    """
    df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    return df


def read_csv_data(df_filename): 
    # Read the CSV file. The file has multi-level headers, hence header=[0, 1].
    df = pd.read_csv(df_filename, header=[0, 1])

    # Drop the first row as it contains only the Date information in one column, which is redundant after setting the index.
    df.drop(index=0, inplace=True)

    # Convert the 'Unnamed: 0_level_0', 'Unnamed: 0_level_1' column (which represents dates) to datetime format.
    # This assumes the dates are in the 'YYYY-MM-DD' format.
    df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')])

    # Set the datetime column as the index of the DataFrame. This makes time series analysis more straightforward.
    df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)

    # Clear the name of the index to avoid confusion, as it previously referred to the multi-level column names.
    df.index.name = None
    return df

def print_data_info(df):
    """
    Print basic information about the DataFrame.
    """
    if df.empty:
        print("DataFrame is empty.")
        return
    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nColumns:")
    print(df.columns)
    print("\nShape:", df.shape)

def print_data_statistics(df):
    """
    Print basic statistics of the DataFrame.
    """
    print("DataFrame Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nUnique Values in Each Column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")

def print_data_head_tail(df, n=5):
    """
    Print the first and last n rows of the DataFrame.
    
    Parameters:
    n (int): Number of rows to print from the start and end of the DataFrame.
    """
    print(f"First {n} rows:")
    print(df.head(n))
    print(f"\nLast {n} rows:")
    print(df.tail(n))

