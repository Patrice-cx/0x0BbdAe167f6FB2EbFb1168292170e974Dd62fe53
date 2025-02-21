import os
import pandas as pd
from datetime import datetime, timedelta

def read_and_combine_csv(folder_path, timezone='Asia/Shanghai'):
    """
    读取指定文件夹中的所有 CSV 文件并合并为一个 DataFrame，
    将 Open time 和 Close time 转换为北京时间，并按 Open time 排序。
    
    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径。
        
    返回:
        pd.DataFrame: 合并后的数据。
    """
    # 列名
    columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
        "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    
    all_data = []  # 用于存储所有 CSV 文件的数据
    
    # 遍历文件夹中的所有 CSV 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):  # 检查文件是否是 CSV 文件
                file_path = os.path.join(root, file)
                print(f"正在读取文件: {file_path}")
                # 读取 CSV 文件
                try:
                    data = pd.read_csv(file_path, header=None, names=columns)
                    if timezone == 'Asia/Shanghai':
                        # 转换 Open time 和 Close time 为北京时间
                        data["Open time"] = data["Open time"].apply(
                            lambda x: datetime.utcfromtimestamp(x / 1000) + timedelta(hours=8)
                        )
                        data["Close time"] = data["Close time"].apply(
                            lambda x: datetime.utcfromtimestamp(x / 1000) + timedelta(hours=8)
                        )
                    elif timezone == 'UTC':
                        # 转换 Open time 和 Close time 为 UTC 时间
                        data["Open time"] = data["Open time"].apply(
                            lambda x: datetime.utcfromtimestamp(x / 1000)
                        )
                        data["Close time"] = data["Close time"].apply(
                            lambda x: datetime.utcfromtimestamp(x / 1000)
                        )
                    
                    all_data.append(data)
                except Exception as e:
                    print(f"读取文件失败: {file_path}, 错误: {e}")
    
    # 合并所有数据
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 根据 Open time 排序
        combined_data = combined_data.sort_values(by="Open time").reset_index(drop=True)
        print(f"数据合并完成并排序，共 {len(combined_data)} 行")
        return combined_data
    else:
        print("未找到任何 CSV 文件")
        return pd.DataFrame(columns=columns)  # 返回空的 DataFrame