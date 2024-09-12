#!/bin/bash

# Read the file list
file_list="/data/home/antoni/datasets/HDTF/filelist_val.txt"

# Get the output folder from the command line argument
output_folder=$1

# Get the keyframes_ckpt from the command line argument, default to none if not provided
keyframes_ckpt=${2:-null}

# Loop through each line in the file list
while IFS= read -r file_name; do
    # Extract the base name without extension
    base_name=$(basename "$file_name" .mp4)

    # Run the Python script with the appropriate arguments
    python scripts/sampling/full_pipeline_keyframes_vid.py \
        --decoding_t 1 \
        --video_path="/fsx/rs2517/data/HDTF/video_crop/${base_name}.mp4" \
        --cond_aug 0. \
        --audio_path="/fsx/rs2517/data/HDTF/audio_emb/${base_name}_wav2vec2_emb.pt" \
        --resize_size=512 \
        --use_latent=True \
        --num_steps=10 \
        --max_seconds=20 \
        --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
        --latent_folder=video_crop_emb \
        --video_folder=video_crop \
        --model_config=scripts/sampling/configs/svd_interpolation_no_bad.yaml \
        --model_keyframes_config=scripts/sampling/configs/svd_keyframes_vid.yaml \
        --get_landmarks=True \
        --landmark_folder=landmarks_crop \
        --overlap=1 \
        --chunk_size=5 \
        --audio_folder=audio \
        --audio_emb_folder=audio_emb \
        --output_folder=/data/home/antoni/results/${output_folder} \
        --keyframes_ckpt=${keyframes_ckpt} \

    echo "Processed $base_name"
done < "$file_list"
