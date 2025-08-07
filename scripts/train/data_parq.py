# -*- coding: utf-8 -*-

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
from tqdm import tqdm
import concurrent.futures
from functools import partial

INPUT_JSON_PATH = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_PT/blip_laion_cc_sbu_558k.json'
IMAGE_BASE_PATH = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_PT/images/'
OUTPUT_PARQUET_DIR = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_PT/parquets_final/'
RECORDS_PER_FILE = 50000
MAX_WORKERS = min(32, os.cpu_count() * 4)


def _write_chunk_to_parquet(records_chunk, output_dir, file_number):
    if not records_chunk: return
    print(f"Writing chunk {file_number} with {len(records_chunk)} records to Parquet...")
    df = pd.DataFrame(records_chunk)
    parquet_schema = pa.schema([
        ('idx', pa.string()), ('image', pa.binary()),
        ('conversations', pa.string()), ('source', pa.string())
    ])
    table = pa.Table.from_pandas(df, schema=parquet_schema, preserve_index=False)
    output_filename = os.path.join(output_dir, f"part_{file_number:04d}.parquet")
    pq.write_table(table, output_filename, compression='snappy')
    print(f"Successfully wrote chunk to: {output_filename}")

def process_single_entry(entry_dict, image_base_path):
    try:
        idx = entry_dict.get("id")
        relative_image_path = entry_dict.get("image")
        conversations = entry_dict.get("conversations")
        source = "blip_laion_cc_sbu"
        if not all([idx, relative_image_path, conversations]): return None
        full_image_path = os.path.join(image_base_path, relative_image_path)
        if os.path.exists(full_image_path):
            with open(full_image_path, 'rb') as img_file:
                image_bytes = img_file.read()
            return {"idx": idx, "image": image_bytes, "conversations": json.dumps(conversations), "source": source}
        return None
    except Exception:
        return None

def process_data_with_controlled_pipeline(json_path, image_base_path, output_dir, max_workers, chunk_size):
    if not os.path.exists(json_path):
        print(f"Error: Input file not found at '{json_path}'")
        return

    print("Loading JSON metadata...")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_entries_metadata = json.load(f)
    total_entries = len(all_entries_metadata)
    print(f"Metadata for {total_entries} entries loaded.")
    os.makedirs(output_dir, exist_ok=True)

    max_tasks_in_flight = max_workers * 4
    worker_function = partial(process_single_entry, image_base_path=image_base_path)
    
    chunk_records, file_counter = [], 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_function, entry) for entry in all_entries_metadata[:max_tasks_in_flight]}
        metadata_iterator = iter(all_entries_metadata[max_tasks_in_flight:])
        
        with tqdm(total=total_entries, desc="Processing All Entries") as pbar:
            while futures:
                done_future = next(concurrent.futures.as_completed(futures))
                futures.remove(done_future)

                record = done_future.result()
                if record:
                    chunk_records.append(record)

                if len(chunk_records) >= chunk_size:
                    _write_chunk_to_parquet(chunk_records, output_dir, file_counter)
                    file_counter += 1
                    chunk_records.clear()

                try:
                    next_entry = next(metadata_iterator)
                    futures.add(executor.submit(worker_function, next_entry))
                except StopIteration:
                    pass

                pbar.update(1)

    if chunk_records:
        _write_chunk_to_parquet(chunk_records, output_dir, file_counter)
    
    print(f"Processing complete. Parquet files are in: {output_dir}")

if __name__ == "__main__":
    process_data_with_controlled_pipeline(
        INPUT_JSON_PATH, 
        IMAGE_BASE_PATH, 
        OUTPUT_PARQUET_DIR, 
        MAX_WORKERS,
        RECORDS_PER_FILE
    )