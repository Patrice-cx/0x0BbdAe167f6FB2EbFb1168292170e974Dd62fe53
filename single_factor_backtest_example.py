import pandas as pd
import numpy as np

from data_reading import read_and_combine_csv
from config import Config
import matplotlib.pyplot as plt

def prepare_data(config):
    fold_name = f'{config.symbols[7]}-spot-klines-15m-from_2018_to_2025'
    df = read_and_combine_csv(f'./raw_data/{fold_name}')
    df.rename({'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, axis=1, inplace=True)
    df['return_15m'] = df['close'].pct_change(1).shift(-1)
    return df

class TillsonT3:
    def __init__(self, length=5, opt=0.1):
        self.length = length
        self.opt = opt

    def calculate(self, data):
        ema1 = data.ewm(span=self.length).mean()
        ema2 = ema1.ewm(span=self.length).mean()
        ema3 = ema2.ewm(span=self.length).mean()
        ema4 = ema3.ewm(span=self.length).mean()
        c1 = -self.opt ** 3
        c2 = 3 * self.opt ** 2 + 3 * self.opt ** 3
        c3 = -6 * self.opt ** 2 - 3 * self.opt - 3 * self.opt ** 3
        c4 = 1 + 3 * self.opt + self.opt ** 3 + 3 * self.opt ** 2
        return c1 * ema4 + c2 * ema3 + c3 * ema2 + c4 * ema1

class TOTT:
    def __init__(self, length=5, opt=0.1, coeff=0.006):
        self.length = length
        self.opt = opt
        self.coeff = coeff

    def calculate(self, data):
        alpha = 2 / (self.length + 1)
        vud1 = np.maximum(data - data.shift(1), 0)
        vdd1 = np.maximum(data.shift(1) - data, 0)
        vUD = vud1.rolling(9).sum()
        vDD = vdd1.rolling(9).sum()
        vCMO = (vUD - vDD) / (vUD + vDD)
        return alpha * np.abs(vCMO) * data + (1 - alpha * np.abs(vCMO)) * data.shift(1)

def williams_r(data, length=3):
    highest_high = data['high'].rolling(length).max()
    lowest_low = data['low'].rolling(length).min()
    return -100 * (highest_high - data['close']) / (highest_high - lowest_low)

def backtest(data, initial_balance=10000):
    balance = initial_balance
    position = 0
    balance_history = []
    t3 = TillsonT3().calculate(data['close'])
    tott = TOTT().calculate(data['close'])
    will_r = williams_r(data)
    
    for i in range(1, len(data)):
        if t3[i] > tott[i] * 1.006 and will_r[i] > -20 and position == 0:
            position = balance / data['close'][i]
            balance = 0
        elif t3[i] < tott[i] * 0.994 and will_r[i] > -70 and position > 0:
            balance = position * data['close'][i]
            position = 0
        balance_history.append(balance + position * data['close'][i])
    
    return balance + position * data['close'].iloc[-1], balance_history

# 读取ETH/USDT数据
config = Config()
df = prepare_data(config)

# avoid overfitting
df = df[df['Open time'] < '2024-06-01']

final_balance, balance_history = backtest(df)

# 绘制收益曲线与资产价格
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['close'], label='ETH/USDT Price', color='blue')
plt.title('ETH/USDT Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df.index[-len(balance_history):], balance_history, label='Portfolio Balance', color='green')
plt.title('Portfolio Balance Over Time')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final Balance: {final_balance}")

