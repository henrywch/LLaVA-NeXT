import pandas as pd
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== CONFIG ======
data_path = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/data/'
image_folder = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/images'
json_out = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/llava70k-v1.6-finetune.json'
num_workers = 8

# ====== LOAD ======
os.makedirs(image_folder, exist_ok=True)

parquet_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# ====== PROCESS ======
def process_row(row):
    if row["image"] is None:
        return None
    
    img_id = row["id"]
    img_bytes = row["image"]['bytes']

    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: image {img_id} cannot be decoded.")
        return None
    save_rel_path = f"images/{img_id}.jpg"
    save_abs_path = os.path.join(image_folder, f"{img_id}.jpg")
    cv2.imwrite(save_abs_path, img)

    record = {
        "id": img_id,
        "image": save_rel_path,
        "conversations": list(row["conversations"])
    }
    return record

output_data = []

# ===== THREADING ======
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    list_futures = []
    for idx, row in df.iterrows():
        list_futures.append(executor.submit(process_row, row))
    for future in tqdm(as_completed(list_futures), total=len(list_futures), desc="Processing samples (multiple threads)"):
        result = future.result()
        if result is not None:
            output_data.append(result)

print(f"Total Entries: {len(output_data)}")

# ===== SAVE ======
with tqdm(total=1, desc="Saving JSON") as pbar:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        pbar.update(1)

print(f"Saved {len(output_data)} records and images.\nJSON: {json_out}")
