# algo-trading-project
1. 如何下载数据？
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

2. 如何训练模型？
    - 运行 python train.py
    - 训练的模型会保存在 ./model/ 目录下

3. 如何测试模型？
    - 运行 python test.py
    - 测试结果会保存在 ./result/ 目录下
