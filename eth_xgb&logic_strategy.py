import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from data_reading import read_and_combine_csv
from feature_engineering import FeatureEngineer
from config import Config

def prepare_data(config, fe):
    fold_name = f'{config.symbols[7]}-spot-klines-15m-from_2018_to_2025'
    df = read_and_combine_csv(f'./raw_data/{fold_name}')
    df.rename({'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, axis=1, inplace=True)
    df = fe.process_data(df)
    df['return_15m'] = df['close'].pct_change(1).shift(-1)
    df['rolling_vol'] = df['return_15m'].rolling(window=1000).std()
    df.dropna(subset=['rolling_vol'], inplace=True)
    config.target_col = 'return_15m'
    config.set_feature_cols(df)
    df['return_class'] = df.apply(lambda row: categorize_return(row), axis=1)
    return df

def categorize_return(row):
    if row['return_15m'] > 1 * row['rolling_vol']:
        return 4
    elif row['return_15m'] > 0.5 * row['rolling_vol']:
        return 3
    elif row['return_15m'] > -0.5 * row['rolling_vol']:
        return 2
    elif row['return_15m'] > -1 * row['rolling_vol']:
        return 1
    else:
        return 0

def train_xgboost(train, config):
    tscv = TimeSeriesSplit(n_splits=5)
    train['predicted_return'] = np.nan
    
    for train_index, val_index in tscv.split(train):
        train_fold, val_fold = train.iloc[train_index], train.iloc[val_index]
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=6)
        model.fit(train_fold[config.feature_cols], train_fold[config.target_col])
        train.loc[val_index, 'predicted_return'] = model.predict(val_fold[config.feature_cols])
    train.dropna(subset=['predicted_return'], inplace=True)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(train[config.feature_cols], train[config.target_col])
    return model, train

def train_logistic_regression(train_scaled, config):
    log_reg_features = config.feature_cols + ['predicted_return']
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, class_weight='balanced')
    model.fit(train_scaled[log_reg_features], train_scaled['return_class'])
    return model

def evaluate_model(test_scaled, predictions):
    print(f"Logistic Regression Accuracy: {accuracy_score(test_scaled['return_class'], predictions):.4f}")
    print("Classification Report:")
    print(classification_report(test_scaled['return_class'], predictions))
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(test_scaled['return_class'], predictions), annot=True, cmap='Blues', fmt='d', xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def trading_strategy_1(test):
    test['position'] = test['predicted_return'].apply(lambda x: 1 if x > 0 else -1)
    test['strategy_return'] = test['position'] * test['return_15m']
    return test

def trading_strategy_2(test_scaled):
    def position_allocating(predicted_class, current_position):
        """
        根据预测类别调整仓位：
        0 - 大跌：空仓或做空
        1 - 下跌：减少持仓
        2 - 震荡：维持现有仓位
        3 - 上涨：增加持仓
        4 - 大涨：全仓或加杠杆
        """
        if predicted_class == 0:
            return -1  # 清仓或做空
        elif predicted_class == 1:
            return max(-1, current_position - 0.5)  # 减少仓位
        elif predicted_class == 2:
            return current_position  # 维持仓位
        elif predicted_class == 3:
            return min(1, current_position + 0.5)  # 增加仓位
        elif predicted_class == 4:
            return 1  # 满仓
        return current_position

    # 应用策略
    test_scaled['position'] = 0  # 初始仓位
    for i in range(1, len(test_scaled)):
        test_scaled.loc[test_scaled.index[i], 'position'] = position_allocating(test_scaled.loc[test_scaled.index[i], 'predicted_class'], test_scaled.loc[test_scaled.index[i-1], 'position'])
    test_scaled['strategy_return'] = test_scaled['position'] * test_scaled['return_15m']

    return test_scaled

def plot_backtest(test):
    (1 + test['strategy_return']).cumprod().plot(figsize=(10,6))
    (1 + test['return_15m']).cumprod().plot(figsize=(10,6))
    plt.show()

def main():
    config = Config()
    fe = FeatureEngineer()
    df = prepare_data(config, fe)
    train, test = train_test_split(df, test_size=0.1, random_state=42, shuffle=False)
    
    # 未经过scaler的数据用于XGBoost训练
    xgb_model, train = train_xgboost(train.copy(), config)
    test['predicted_return'] = xgb_model.predict(test[config.feature_cols])
    
    # 标准化数据用于Logistic Regression
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[config.feature_cols] = scaler.fit_transform(train[config.feature_cols])
    test_scaled[config.feature_cols] = scaler.transform(test[config.feature_cols])
    
    log_reg_model = train_logistic_regression(train_scaled, config)
    test_scaled['predicted_class'] = log_reg_model.predict(test_scaled[config.feature_cols + ['predicted_return']])
    
    evaluate_model(test_scaled, test_scaled['predicted_class'])

    test = trading_strategy_1(test)
    plot_backtest(test)

    test_scaled = trading_strategy_2(test_scaled)
    plot_backtest(test_scaled)
    
if __name__ == "__main__":
    main()
