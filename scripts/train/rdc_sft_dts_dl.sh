#!/bin/bash

set -e

# --- Configuration ---
readonly REPO_ID="lmms-lab/LLaVA-NeXT-Data"
readonly TARGET_DIR="/root/autodl-tmp/datasets/LLaVA_Train/LLaVA_SFT/"
readonly SAMPLE_SIZE=25
readonly TOTAL_FILES=250
readonly FILE_PREFIX="data/train"
readonly FILE_SUFFIX="-of-00250.parquet"
readonly PARALLEL_JOBS=4

# --- Main Script ---
echo "Generating potential file list based on known naming pattern..."

mapfile -t selected_files < <(
    for i in $(seq 0 $((TOTAL_FILES - 1))); do
        printf "${FILE_PREFIX}-%05d${FILE_SUFFIX}
" "$i"
    done | shuf -n "$SAMPLE_SIZE"
)

if [ ${#selected_files[@]} -eq 0 ]; then
    echo "Error: Failed to generate and sample the file list. Exiting."
    exit 1
fi

echo "Randomly selected ${#selected_files[@]} files. Starting parallel download..."

mkdir -p "$TARGET_DIR"
echo "Files will be saved to: $TARGET_DIR"
echo "Running up to ${PARALLEL_JOBS} downloads simultaneously."

printf "%s
" "${selected_files[@]}" | xargs -n 1 -P $PARALLEL_JOBS -I {} \
bash -c '
    file="{}"
    echo "--------------------------------------------------"
    echo "Starting download for: $file"
    huggingface-cli download \
        --repo-type dataset \
        "'$REPO_ID'" \
        "$file" \
        --local-dir "'$TARGET_DIR'" \
        --resume-download
    echo "--> Successfully downloaded: $file"
'

echo "--------------------------------------------------"
echo "All done. All files are in $TARGET_DIR"
