#!/bin/bash

# Read the file list


# Get the output folder from the command line argument
output_folder=$1

file_list=${2:-"/data/home/antoni/datasets/HDTF/filelist_val.txt"}

# Get the keyframes_ckpt from the command line argument, default to none if not provided
keyframes_ckpt=${3:-None}

# Get the interpolation_ckpt from the command line argument, default to none if not provided
interpolation_ckpt=${4:-None}

overlapping=${5:-1}

# Check if keyframes_ckpt is provided and not null
# if [ "$keyframes_ckpt" != "null" ]; then
#     # Extract the folder name from the path
#     folder_name=$(echo "$keyframes_ckpt" | sed -n 's/.*logs\/\([0-9T-]*_[^\/]*\).*/\1/p')

    
#     # Create the destination directory if it doesn't exist
#     mkdir -p checkpoints/infered_models
    
#     # Copy the checkpoint file to the new location with the new name
#     cp "$keyframes_ckpt" "checkpoints/infered_models/${folder_name}.pt" || { echo "Failed to copy keyframes checkpoint"; exit 1; }
    
#     echo "Copied keyframes checkpoint to checkpoints/infered_models/${folder_name}.pt"
    
#     # Update keyframes_ckpt to use the new path
#     keyframes_ckpt="checkpoints/infered_models/${folder_name}.pt"
# fi

# output_dir="/data/home/antoni/results/${output_folder}"
# if [ -d "$output_dir" ]; then
#     last_video=$(ls -v "$output_dir"/*.mp4 2>/dev/null | tail -n 1)
#     if [ -n "$last_video" ]; then
#         starting_index=$(basename "$last_video" .mp4)
#         starting_index=$((10#$starting_index + 1))
#     else
#         starting_index=None
#     fi
# else
#     starting_index=None
# fi

# echo "Starting index: $starting_index"

# Run the Python script with the appropriate arguments
python scripts/sampling/full_pipeline_paper.py \
    --filelist=${file_list} \
    --decoding_t 1 \
    --cond_aug 0. \
    --resize_size=512 \
    --use_latent=True \
    --max_seconds=14 \
    --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
    --latent_folder=video_crop_emb \
    --video_folder=video_crop \
    --model_config=scripts/sampling/configs/svd_interpolation_high_quali.yaml \
    --model_keyframes_config=scripts/sampling/configs/svd_keyframes_no_cross.yaml \
    --get_landmarks=False \
    --landmark_folder=landmarks_crop \
    --overlap=${overlapping} \
    --chunk_size=2 \
    --audio_folder=audio \
    --audio_emb_folder=audio_emb \
    --output_folder=/data/home/antoni/results_updated/${output_folder} \
    --keyframes_ckpt=${keyframes_ckpt} \
    --interpolation_ckpt=${interpolation_ckpt} \
    --double_first=False \
    --add_zero_flag=True \
    --emotion_folder=emotions \
    --extra_audio=both \
    --compute_until=45 \
    --audio_emb_type=wavlm \
    # --starting_index=${starting_index} \

