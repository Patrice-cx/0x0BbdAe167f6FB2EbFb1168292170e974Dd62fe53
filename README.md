# 如何下载数据？

## 下载量价数据
- 运行 python download_data.py
- 下载的文件会保存在 ./row_data/ 目录下
- 程序参数包括：
    - --spot_future 选择下载现货数据（spot）或是期货数据（future）
    - --data_type 选择下载数据类型 klines（K线数据）或是tick（逐笔数据）
    - --frequency 选择下载数据的batch monthly（月度数据）
    - --start_year 下载数据的开始年份
    - --end_date 下载数据的结束年份
    - --interval 下载数据的间隔，例如 15m，30m，1h 等
    - --crypto_pairs 选择下载数据的加密货币对，例如 BTCUSDT，ETHUSDT 等

## 下载链上数据
- 运行 python OnChain_data_downloading.py
- 下载的文件会保存在 ./blockchair_data_blocks/ 目录下

# 如何读取数据？

- from data_reading import read_and_combine_csv
- read_and_combine_csv 函数会读取 ./row_data/ 目录下的 csv 文件，并将它们合并成一个 DataFrame
- 返回的 DataFrame 包含了以下列：
    - "Open time", 开盘时间
    - "Open", 开盘价格
    - "High", 最高价
    - "Low", 最低价
    - "Close", 收盘价
    - "Volume", 成交量
    - "Close time", 收盘时间
    - "Quote asset volume",  成交额
    - "Number of trades", 成交笔数
    - "Taker buy base asset volume", 买方成交量
    - "Taker buy quote asset volume", 买方成交额
    - "Ignore", 忽略字段

# 什么是config.py?

- config.py 文件包含了程序的一些配置信息，例如 随机数种子
- config.py 文件还包含了一些全局变量，例如 训练特征 目标特征等 可以在模型、策略代码中初始化

# 特征构造/因子挖掘/feature_engineering.py

- 该文件包含了特征构造的函数，例如：
    - trend_indicators 趋势指标
    - momentum_indicators 动量指标
    - volume_indicators 量价指标
    - custom_oscilators 自定义震荡指标
    - composite_indicators 综合指标
    - volatility_indicators 波动率指标
    - create_datetime_features 创建时间特征
- 更多有待挖掘的因子
    - 各种技术指标
    - 各种统计指标
    - 时序或是截面的
    - 订单薄
    - 链上信息
    - 市场情绪，舆情，新闻
    - 衍生品数据
    - 宏观经济
    - 机构数据

# single_factor_backtest_example.py

- 该文件提供了单因子回测的代码的示例
- 在验证单因子的有效性的时候须注意，不要使用后续模型训练预测期间所使用的预测时间段的数据，以避免模型过拟合

# display.py

- 该文件包含了用于显示数据的函数
- 代码包含    start_date = "2021-03-06" 以及 end_date = "2021-03-07"； 通过修改这两个参数可以plot不同时间段内的价格k线走势（注意最好plot的时间段不超过两天，不然显示不好）
- 文件用于观察任意时间段内交易对的价格走势，验证因子构造猜想

# 策略_1/eth_xgb&logic_strategy.py

- 该文件包含了模型和策略的代码
- 模型包括使用XGBoost进行对收益率的回归预测，然后将预测值进一步输送给logistic 进行分类
- 策略包括：（1）使用XGBoost的回归预测直接进行交易 （2）使用logistic分类进行交易

