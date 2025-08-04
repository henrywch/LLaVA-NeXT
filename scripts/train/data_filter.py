import json
import argparse
import random

def filter_data(records: list) -> list:
    """
    Filters a list of dictionaries, removing items where the 'image' key's
    value contains the substring 'ocr_vqa'.

    Args:
        records (list): A list of dictionary objects (your dataset).

    Returns:
        list: A new list containing only the filtered records.
    """
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=60000,
        help="Number of samples to randomly select from the filtered data. Default is 60000."
    )
    
    args = parser.parse_args()

    try:
        print(f"Attempting to read data from '{args.input_file}'...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Could not parse as a single JSON array. Treating as JSON Lines format.")
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]
        
        if not isinstance(data, list):
            print("Error: The loaded data is not a list of records. Aborting.")
            return

        print(f"Successfully loaded {len(data)} records.")

        filtered_records = filter_data(data)
        print(f"Filtering complete. {len(filtered_records)} records remain.")

        if filtered_records:
            if len(filtered_records) > args.num_samples:
                sample_size = args.num_samples
                print(f"Randomly sampling {int(sample_size / 1000)}k records from the filtered dataset...")
            else:
                print(f"Total Entries: {len(filtered_records)}\nNo need to sample.")
                exit(0)

            sampled_records = random.sample(filtered_records, k=sample_size)
            print(f"Sampling complete. {len(sampled_records)} records were selected.")
        else:
            print("Filtered dataset is empty. No sampling will be performed.")
            sampled_records = []

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_records, f, indent=2, ensure_ascii=False)
            
        print(f"Final sampled data has been saved to '{args.output_file}'.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except ValueError as ve:
        print(f"Error during sampling: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
