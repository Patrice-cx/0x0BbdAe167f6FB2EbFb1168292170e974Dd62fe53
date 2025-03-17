import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta

def generate_date_list(start_date, end_date):
    date_list = []
    current_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_list

DATA_DIR = "blockchair_data_blocks"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://gz.blockchair.com/ethereum/blocks/"
start_date = "20241201"
end_date = "20241231"
Timestamps = generate_date_list(start_date, end_date)
datasets = []
for timestamp in Timestamps:
    datasets.append(f"blockchair_ethereum_blocks_{timestamp}")


def download_and_extract(dataset):
    url = f"{BASE_URL}{dataset}.tsv.gz"
    local_gz_path = os.path.join(DATA_DIR, f"{dataset}.tsv.gz")
    local_tsv_path = os.path.join(DATA_DIR, f"{dataset}.tsv")

    print(f"Downloading {dataset} data...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(local_gz_path, 'wb') as f:
            f.write(response.content)
        print(f"Download complete: {local_gz_path}")

        print(f"Extracting {dataset} data...")
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(local_tsv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Extraction complete: {local_tsv_path}")

        # Optionally, delete the .gz file after extraction
        os.remove(local_gz_path)

    else:
        print(f"Failed to download {dataset}. Status code: {response.status_code}")

# Download and extract all datasets
for dataset in datasets:
    download_and_extract(dataset)

print("All datasets downloaded and extracted successfully!")