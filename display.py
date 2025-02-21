import pandas as pd
import mplfinance as mpf
from data_reading import read_and_combine_csv
from config import Config

def load_and_prepare_data(folder_path, start_date, end_date):
    """
    Load and filter data for a specified date range.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Filtered DataFrame with datetime index and necessary columns.
    """
    # Load and combine CSV files
    df = read_and_combine_csv(folder_path, 'UTC')

    # Ensure the "Open time" column is in datetime format
    df["Open time"] = pd.to_datetime(df["Open time"])

    # Filter the data within the date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include the end date fully

    filtered_df = df[(df["Open time"] >= start_dt) & (df["Open time"] < end_dt)]

    return filtered_df

def prepare_mpf_data(df):
    """
    Prepare data for mplfinance plotting.

    Args:
        df (pd.DataFrame): Filtered DataFrame with OHLC and Volume columns.

    Returns:
        pd.DataFrame: Data formatted for mplfinance.
    """
    return df.set_index("Open time")[["Open", "High", "Low", "Close", "Volume"]].astype(float)

def plot_candlestick_chart(mpf_df, title):
    """
    Plot a candlestick chart using mplfinance.

    Args:
        mpf_df (pd.DataFrame): Data formatted for mplfinance.
        title (str): Title of the chart.
    """
    mpf.plot(
        mpf_df,
        type="candle",
        title=title,
        volume=True,
        style="yahoo",
        datetime_format="%Y-%m-%d %H:%M"
    )

def main():
    """
    Main function to execute the workflow.
    """
    # Load configurations
    config = Config()
    folder_name = f'ETHUSDT-spot-klines-15m-from_2020_to_2023'
    folder_path = f'.//raw_data//{folder_name}'

    # Define date range
    start_date = "2021-03-06"
    end_date = "2021-03-07"

    # try:
    # Load and filter data
    filtered_df = load_and_prepare_data(folder_path, start_date, end_date)

    # Prepare data for plotting
    mpf_df = prepare_mpf_data(filtered_df)

    # Plot the candlestick chart
    title = f"BTC/USDT Candlestick Chart ({start_date} to {end_date})"
    plot_candlestick_chart(mpf_df, title)

    # except Exception as e:
    #     print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()