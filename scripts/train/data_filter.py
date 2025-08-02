import json
import argparse
import random  # 新增：导入 random 模块用于随机抽样

def filter_data(records: list) -> list:
    """
    Filters a list of dictionaries, removing items where the 'image' key's
    value contains the substring 'ocr_vqa'.

    Args:
        records (list): A list of dictionary objects (your dataset).

    Returns:
        list: A new list containing only the filtered records.
    """
    # We keep an item if the removal condition is NOT met.
    # The removal condition is: 'image' key exists AND 'ocr_vqa' is in its value.
    # A list comprehension is a concise way to do this.
    return [
        item for item in records
        if not ('image' in item and 'ocr_vqa' in item.get('image', ''))
    ]

def main():
    """
    Main function to load, process, and save the dataset.
    """
    parser = argparse.ArgumentParser(
        description="Filter a JSON or JSONL dataset to remove records where the 'image' key contains 'ocr_vqa' and then randomly sample a portion of the result."
    )
    parser.add_argument("input_file", help="Path to the input JSON or JSONL file.")
    parser.add_argument("output_file", help="Path for the filtered and sampled output JSON file.")
    # 新增：添加一个可选参数来控制抽样比例，默认为 10%
    parser.add_argument(
        "--sample_ratio", 
        type=float, 
        default=0.1, 
        help="Fraction of the filtered data to randomly sample (e.g., 0.1 for 10%%). Default is 0.1."
    )
    
    args = parser.parse_args()

    try:
        print(f"Attempting to read data from '{args.input_file}'...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            # Try to load as a standard JSON array first
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If that fails, assume it's a JSON Lines (.jsonl) file
                print("Could not parse as a single JSON array. Treating as JSON Lines format.")
                f.seek(0)  # Go back to the start of the file
                # Read each line, parse as JSON, and ignore empty lines
                data = [json.loads(line) for line in f if line.strip()]
        
        if not isinstance(data, list):
            print("Error: The loaded data is not a list of records. Aborting.")
            return

        print(f"Successfully loaded {len(data)} records.")

        # 1. 执行过滤
        filtered_records = filter_data(data)
        print(f"Filtering complete. {len(filtered_records)} records remain.")

        # 2. 新增：执行随机抽样
        if filtered_records:
            # 计算抽样数量
            sample_size = int(len(filtered_records) * args.sample_ratio)
            print(f"Randomly sampling {sample_size} records ({args.sample_ratio:.0%}) from the filtered dataset...")
            
            # 使用 random.sample 进行无放回的随机抽样
            sampled_records = random.sample(filtered_records, k=sample_size)
            print(f"Sampling complete. {len(sampled_records)} records were selected.")
        else:
            print("Filtered dataset is empty. No sampling will be performed.")
            sampled_records = []

        # 3. 保存最终抽样的结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            # We'll save the output as a standard, pretty-printed JSON file for readability.
            json.dump(sampled_records, f, indent=2, ensure_ascii=False) # 添加 ensure_ascii=False 以支持中文等非-ASCII字符
            
        print(f"Final sampled data has been saved to '{args.output_file}'.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except ValueError as ve:
        # 捕捉 random.sample 可能抛出的错误，例如当抽样数量大于总体数量时
        print(f"Error during sampling: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
