from huggingface_hub import HfApi, HfFolder
import os

repo_id = "henrywch2huggingface/llavanext-scaled-0.5b"

upload_mappings = [
    {
        "local_path": "/root/autodl-tmp/models/projectors/llavanext-_root_autodl-tmp_models_siglip-so400m-p14-384-_root_autodl-tmp_models_qwen2.5-0.5b-instr-mlp2x_gelu-pretrain_blip558k_plain",
        "repo_path": "projectors"
    },
    {
        "local_path": "/root/autodl-tmp/models/llavanext-mid-0.5b",
        "repo_path": "transition_model"
    },
    {
        "local_path": "/root/autodl-tmp/models/llavanext-scaled-0.5b/training_loss.png",
        "repo_path": "training_loss.png"
    }
]

api = HfApi()

print(f"Creating repository '{repo_id}' if it doesn't exist...")
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

print("Starting to upload files...")
for mapping in upload_mappings:
    local_file_or_dir = mapping["local_path"]
    path_in_repo = mapping["repo_path"]

    if not os.path.exists(local_file_or_dir):
        print(f"⚠️  Warning: Local file not found, skipping: {local_file_or_dir}")
        continue
    
    try:
        if os.path.isdir(local_file_or_dir):
            print(f"  - Uploading DIRECTORY '{local_file_or_dir}' to '{path_in_repo}'...")
            api.upload_folder(
                folder_path=local_file_or_dir,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model"
            )
        elif os.path.isfile(local_file_or_dir):
            print(f"  - Uploading FILE '{local_file_or_dir}' to '{path_in_repo}'...")
            api.upload_file(
                path_or_fileobj=local_file_or_dir,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model"
            )
    except Exception as e:
        print(f"❌  Error uploading {local_file_or_dir}: {e}")

print(f"✅  Upload complete! Check your dataset at: https://huggingface.co/datasets/{repo_id}/tree/main")
