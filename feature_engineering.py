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
        self.windows = [4,8, 48, 96,480, 960]

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
        #andrew
        # Aroon Indicator - Common window: 25
        aroon = AroonIndicator(high=df['high'],low=df['low'], window=24)
        df['aroon_up_24'] = aroon.aroon_up()
        df['aroon_down_24'] = aroon.aroon_down()
        df['aroon_indicator_24'] = aroon.aroon_indicator()


        # 9. Commodity Channel Index (CCI) - Common window: 20
        df['cci_24'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=24).cci()

        #andrew

        return df

    def momentum_indicators(self, df):
        """Create momentum indicators."""
        for length in self.windows:
            # 7. Relative Strength Index (RSI)
            df[f'rsi_{length}'] = RSIIndicator(close=df['close'], window=length).rsi()

        # andrew
        #Williams % R
        df[f'williamsr_{8}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                           lbp=8).williams_r()

        df[f'williamsr_{24}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                  lbp=96).williams_r()

        df[f'williamsr_{48}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                  lbp=96).williams_r()

        df[f'williamsr_{96}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                  lbp=96).williams_r()

        # Stochastic RSI - Common window: 14, smoothed with 3 periods
        stoch_rsi = StochRSIIndicator(close=df['close'], window=16, smooth1=3, smooth2=3)
        df['stoch_rsi_16'] = stoch_rsi.stochrsi()
        df['stoch_rsi_k_16'] = stoch_rsi.stochrsi_k()
        df['stoch_rsi_d_16'] = stoch_rsi.stochrsi_d()

        # True Strength Index (TSI) - Common slow=25, fast=13
        df['tsi_24_12'] = TSIIndicator(close=df['close'], window_slow=24, window_fast=12).tsi()

        # Awesome Oscillator - Uses (5, 34) as default
        df['awesome_oscillator'] = AwesomeOscillatorIndicator(high=df['high'], low=df['low'], window1=4,
                                                              window2=32).awesome_oscillator()

        # andrew
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
        # andrew
        # 好像挺有用的
        for length in [4*2,4*24]:
            df[f'vollume_{length}p_quantile']=df['volume'].rolling(length,min_periods=int(length/2)).apply(lambda x:x.rank(pct=True).iloc[-1],raw=False)
        # andrew
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


        # andrew
        # astype categoryy也会跑不动
        # df['Year'] = df['Year'].astype('category')
        # df['Month'] = df['Month'].astype('category')
        # df['Day'] = df['Day'].astype('category')
        # df['Open hour'] = df['Open hour'].astype('category')
        #
        # df['Weekday'] = df['Weekday'].astype('category')
        # andrew

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

class FeatureEngineeringRL:
    def __init__(self):
        pass

    def process_data(self, df):
        # MACD - Moving Average Convergence Divergence
        df['MACD'] = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        df['MACD_SIGNAL'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)

        # RSI - Relative Strength Index
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # VWAP - Volume Weighted Average Price
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])

        # EMA - Exponential Moving Average
        df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)

        # Bollinger Bands
        df['BB_MIDDLE'] = ta.trend.sma_indicator(df['Close'], window=20)  # Middle Band (SMA)
        df['BB_UPPER'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)  # Upper Band
        df['BB_LOWER'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)  # Lower Band
        return df