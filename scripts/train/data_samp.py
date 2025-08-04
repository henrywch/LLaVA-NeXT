import json
import random

input_json_path = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_S1_5/coco118k_stage1.5_finetune_w_prompt.json'
output_json_path = '/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_S1_5/coco30k_stage1.5_finetune_w_prompt.json'
sample_size = 30000

def main():
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")

    sampled_data = random.sample(data, sample_size)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sampled_data)} sampled records to {output_json_path}.")

if __name__ == "__main__":
    main()
