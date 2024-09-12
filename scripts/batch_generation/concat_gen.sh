#!/bin/bash

# Read the file list
file_list="/data/home/antoni/datasets/HDTF/filelist_val.txt"

# Loop through each line in the file list
while IFS= read -r file_name; do
    # Extract the base name without extension
    base_name=$(basename "$file_name" .mp4)

    # Run the Python script with the appropriate arguments
    python scripts/sampling/full_pipeline_overlap.py \
        --decoding_t 2 \
        --video_path="/fsx/rs2517/data/HDTF/video_crop/${base_name}.mp4" \
        --cond_aug 0. \
        --audio_path="/fsx/rs2517/data/HDTF/audio_emb/${base_name}_wav2vec2_emb.pt" \
        --resize_size=512 \
        --use_latent=True \
        --num_steps=10 \
        --max_seconds=12 \
        --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
        --latent_folder=video_crop_emb \
        --video_folder=video_crop \
        --model_config=scripts/sampling/configs/svd_interpolation.yaml \
        --model_keyframes_config=scripts/sampling/configs/svd_image.yaml \
        --get_landmarks=True \
        --landmark_folder=landmarks_crop \
        --overlap=1 \
        --chunk_size=5 \
        --audio_folder=audio \
        --audio_emb_folder=audio_emb \
        --output_folder=/data/home/antoni/results/image_cat_overlap_1 \

    echo "Processed $base_name"
done < "$file_list"
