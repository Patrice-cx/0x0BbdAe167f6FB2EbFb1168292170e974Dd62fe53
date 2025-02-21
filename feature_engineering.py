from config import Config

# feature_engineering.py
import pandas as pd
import numpy as np
import ta

from ta.trend import EMAIndicator, SMAIndicator, WMAIndicator, ADXIndicator, CCIIndicator, MACD, AroonIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator, WilliamsRIndicator, StochRSIIndicator, TSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, AccDistIndexIndicator, ForceIndexIndicator, EaseOfMovementIndicator
from ta.others import DailyLogReturnIndicator, CumulativeReturnIndicator


class FeatureEngineer:
    """Feature engineering class for creating technical indicators and other features."""
    def __init__(self):
        self.windows = [8, 48, 96]

    def trend_indicators(self, df):
        """Create trend indicators."""
        for length in self.windows:
            # Exponential Moving Average (EMA)
            df[f'ema_{length}'] = EMAIndicator(close=df['close'], window=length).ema_indicator()

        # Moving Average Convergence Divergence (MACD)
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df

    def momentum_indicators(self, df):
        """Create momentum indicators."""
        for length in self.windows:
            # 7. Relative Strength Index (RSI)
            df[f'rsi_{length}'] = RSIIndicator(close=df['close'], window=length).rsi()
        return df

    def volatility_indicators(self, df):
        """Create volatility indicators."""
        for length in self.windows:
            # 10. Bollinger Bands
            bb = BollingerBands(close=df['close'], window=length, window_dev=2)
            df[f'bb_upper_{length}'] = bb.bollinger_hband()
            df[f'bb_lower_{length}'] = bb.bollinger_lband()
            df[f'bb_width_{length}'] = bb.bollinger_wband()

            # 11. Average True Range (ATR)
            df[f'atr_{length}'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=length).average_true_range()

            # 12. Donchian Channels
            dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=length)
            df[f'donchian_high_{length}'] = dc.donchian_channel_hband()
            df[f'donchian_low_{length}'] = dc.donchian_channel_lband()
        return df

    def volume_indicators(self, df):
        """Create volume indicators."""
        # 13. On-Balance Volume (OBV)
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        return df

    def custom_oscilators(self, df):
        """Create custom oscillators."""
        for length in self.windows:
            # 16. Rate of Change (ROC)
            df[f'roc_{length}'] = ROCIndicator(close=df['close'], window=length).roc()
        return df

    def composite_indicators(self, df):
        """Create composite indicators."""
        # 19. Daily Log Return
        df['daily_log_return'] = DailyLogReturnIndicator(close=df['close']).daily_log_return()

        # 20. Cumulative Returns
        df['cumulative_return'] = CumulativeReturnIndicator(close=df['close']).cumulative_return()
        return df
    
    @staticmethod
    def create_datetime_features(df):
        """Create datetime-based features."""
        df['Year'] = df['Open time'].dt.year
        df['Month'] = df['Open time'].dt.month
        df['Day'] = df['Open time'].dt.day
        df['Open hour'] = df['Open time'].dt.hour
        return df

    def process_data(self, df):
        """Process the data for model training."""
        df = self.momentum_indicators(df)
        df = self.volume_indicators(df)
        df = self.trend_indicators(df)
        df = self.custom_oscilators(df)
        df = self.composite_indicators(df)
        df = self.volatility_indicators(df)
        df = self.create_datetime_features(df)
        return df
