import subprocess
import os
import argparse

def download_data(base_url, start_year, end_year, months, download_folder, crypto_pairs, interval, spot_future):
    # 循环下载、解压并删除 ZIP 文件
    for year in range(start_year, end_year + 1):
        for month in months:
            # 格式化月份为两位数
            month_str = f"{month:02d}"
            # 构建文件名和 URL
            file_name = f"{crypto_pairs}-{interval}-{year}-{month_str}.zip"
            file_path = os.path.join(download_folder, file_name)  # 文件的完整路径
            url = f"{base_url}{file_name}"
            
            try:
                # 下载文件到指定文件夹
                print(f"正在下载: {url}")
                subprocess.run(["curl", "-s", url, "-o", file_path], check=True)
                print(f"下载完成: {file_path}")
                
                # 解压文件到当前目录
                print(f"正在解压: {file_path}")
                subprocess.run(["unzip", "-o", file_path, "-d", download_folder], check=True)
                print(f"解压完成: {file_path}")
                
                # 删除 ZIP 文件
                print(f"正在删除 ZIP 文件: {file_path}")
                os.remove(file_path)
                print(f"删除完成: {file_path}")
            
            except subprocess.CalledProcessError as e:
                print(f"操作失败: {e}")
            except FileNotFoundError:
                print(f"文件未找到，无法删除: {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spot_future', type = str, default = 'spot')
    parser.add_argument('--data_type', type = str, default = 'klines')
    parser.add_argument('--frequency', type = str, default = 'monthly')
    parser.add_argument('--interval', type = str, default = '15m')
    parser.add_argument('--start_year', type = int, default = 2018)
    parser.add_argument('--end_year', type = int, default = 2025)
    parser.add_argument('--crypto_pairs', type = str, default = 'ETHUSDT')
    args = parser.parse_args()

    # 定义年份和月份范围
    start_year = args.start_year
    end_year = args.end_year
    months = range(1, 13)

    # 定义下载存储的文件夹
    download_folder = f"./raw_data/{args.crypto_pairs}-{args.spot_future}-{args.data_type}-{args.interval}-from_{args.start_year}_to_{args.end_year}"

    # 创建文件夹（如果不存在）
    os.makedirs(download_folder, exist_ok=True)

    # 基础 URL
    if args.spot_future == 'spot':
        base_url = f"https://data.binance.vision/data/{args.spot_future}/{args.frequency}/{args.data_type}/{args.crypto_pairs}/{args.interval}/"
    elif args.spot_future == 'futures':
        base_url = f"https://data.binance.vision/data/{args.spot_future}/um/{args.frequency}/{args.data_type}/{args.crypto_pairs}/{args.interval}/"
    download_data(base_url, start_year, end_year, months, download_folder, args.crypto_pairs, args.interval, args.spot_future)