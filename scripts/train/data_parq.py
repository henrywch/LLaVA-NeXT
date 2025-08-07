# -*- coding: utf-8 -*-

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
from tqdm import tqdm
import concurrent.futures
from functools import partial
import ijson

INPUT_JSON_PATH = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/llava_v1_5_mix60k.json'
IMAGE_BASE_PATH = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/'
OUTPUT_PARQUET_DIR = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/.parquets/'
RECORDS_PER_FILE = 50000
MAX_WORKERS = min(32, os.cpu_count() * 4)
INTERNAL_BATCH_SIZE = 1000 

def process_single_entry(entry_dict, image_base_path):
    """
    Processes a single JSON entry: reads the image and returns a structured dictionary.
    """
    try:
        idx = entry_dict.get("id")
        relative_image_path = entry_dict.get("image")
        conversations = entry_dict.get("conversations")
        source = "coco"
        if not all([idx, relative_image_path, conversations]):
            return None

        full_image_path = os.path.join(image_base_path, relative_image_path)
        if os.path.exists(full_image_path):
            with open(full_image_path, 'rb') as img_file:
                image_bytes = img_file.read()

            return {
                "idx": str(idx), 
                "image": image_bytes, 
                "conversations": json.dumps(conversations), 
                "source": source
            }
        return None
    except Exception:
        return None

def rename_final_files(output_dir):
    """
    Renames the generated parquet files to include the total file count.
    Example: part_0000.parquet -> part_0000_of_12.parquet
    """
    print("All initial parquet files generated. Starting renaming process...")
    try:
        initial_files = sorted([f for f in os.listdir(output_dir) if f.startswith('part_') and f.endswith('.parquet') and '_of_' not in f])
        
        if not initial_files:
            print("No files found to rename or files are already renamed.")
            return
            
        total_num_of_files = len(initial_files)
        print(f"Found {total_num_of_files} parquet files. Renaming now...")

        for index, filename in enumerate(initial_files):
            old_path = os.path.join(output_dir, filename)
            new_filename = f"part_{index:04d}_of_{total_num_of_files:04d}.parquet"
            new_path = os.path.join(output_dir, new_filename)
            os.rename(old_path, new_path)
            
        print(f"Successfully renamed {total_num_of_files} files.")

    except FileNotFoundError:
        print(f"Error: Output directory not found at '{output_dir}'")
    except Exception as e:
        print(f"An error occurred during renaming: {e}")


def process_data_with_full_streaming(json_path, image_base_path, output_dir, max_workers, records_per_file, internal_batch_size):
    """
    Main function to process data using full streaming for both JSON input and Parquet output.
    """
    if not os.path.exists(json_path):
        print(f"Error: Input file not found at '{json_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    parquet_schema = pa.schema([
        ('idx', pa.string()),
        ('image', pa.binary()),
        ('conversations', pa.string()),
        ('source', pa.string())
    ])
    print("Parquet schema defined and will be applied to all output files.")

    worker_function = partial(process_single_entry, image_base_path=image_base_path)
    
    writer = None
    file_counter = 0
    records_in_current_file = 0
    batch_records = []

    print("Starting stream processing from JSON...")
    with open(json_path, 'r', encoding='utf-8') as f, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        metadata_iterator = ijson.items(f, 'item')
        processed_records_iterator = executor.map(worker_function, metadata_iterator, chunksize=max_workers*2)

        for record in tqdm(processed_records_iterator, desc="Processing Entries", unit=" records"):
            if not record:
                continue

            if writer is None:
                output_filename = os.path.join(output_dir, f"part_{file_counter:04d}.parquet")
                writer = pq.ParquetWriter(output_filename, parquet_schema, compression='snappy')

            batch_records.append(record)

            if len(batch_records) >= internal_batch_size:
                df_batch = pd.DataFrame(batch_records)
                table_batch = pa.Table.from_pandas(df_batch, schema=parquet_schema, preserve_index=False)
                writer.write_table(table_batch)
                records_in_current_file += len(batch_records)
                batch_records = []

            if records_in_current_file >= records_per_file:
                if batch_records:
                    df_batch = pd.DataFrame(batch_records)
                    table_batch = pa.Table.from_pandas(df_batch, schema=parquet_schema, preserve_index=False)
                    writer.write_table(table_batch)
                    records_in_current_file += len(batch_records)
                    batch_records = []

                print(f"Successfully wrote {records_in_current_file} records to: {output_filename}")
                writer.close()
                writer = None
                file_counter += 1
                records_in_current_file = 0

    if writer and batch_records:
        df_batch = pd.DataFrame(batch_records)
        table_batch = pa.Table.from_pandas(df_batch, schema=parquet_schema, preserve_index=False)
        writer.write_table(table_batch)
        records_in_current_file += len(batch_records)
        print(f"Successfully wrote {records_in_current_file} records to: {output_filename}")
        writer.close()

    rename_final_files(output_dir)
    
    print(f"Processing complete. Final Parquet files are in: {output_dir}")

if __name__ == "__main__":
    process_data_with_full_streaming(
        INPUT_JSON_PATH, 
        IMAGE_BASE_PATH, 
        OUTPUT_PARQUET_DIR, 
        MAX_WORKERS,
        RECORDS_PER_FILE,
        INTERNAL_BATCH_SIZE
    )
