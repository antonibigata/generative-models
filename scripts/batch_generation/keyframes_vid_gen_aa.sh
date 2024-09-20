#!/bin/bash

# Read the file list
file_list="/data/home/antoni/datasets/filelist_video_aa_val.txt"

# Get the output folder from the command line argument
output_folder=$1

# Get the keyframes_ckpt from the command line argument, default to none if not provided
keyframes_ckpt=${2:-null}

overlapping=${3:-1}

# Loop through each line in the file list
while IFS= read -r file_name; do

    # Increment the counter
    ((counter++))

    # Check if the counter is a multiple of 10
    if (( counter % 10 != 0 )); then
        # Skip this iteration if it's not a multiple of 10
        continue
    fi
    # Extract the base name without extension
    # Extract the directory part (up to video_crop)
    dir_part=$(dirname "$file_name")
    audio_dir_part=$(echo "$dir_part" | sed 's/video_crop/audio_emb/')
    emotion_dir_part=$(echo "$dir_part" | sed 's/video_crop/emotions/')

    # Extract the filename without extension and output suffix
    file_part=$(basename "$file_name" | sed -E 's/_output_output\.mp4$//')

    counter=0

    # Skip if file does not exist
    if [ ! -f "/fsx/behavioural_computing_data/face_generation_data/AA_processed/${emotion_dir_part}/${file_part}_output_output.pt" ]; then
        echo "Skipping /fsx/behavioural_computing_data/face_generation_data/AA_processed/${emotion_dir_part}/${file_part}_output_output.pt because it does not exist"
        continue
    fi


    # Run the Python script with the appropriate arguments
    python scripts/sampling/full_pipeline_keyframes_vid.py \
        --decoding_t 1 \
        --video_path="/fsx/behavioural_computing_data/face_generation_data/AA_processed/${dir_part}/${file_part}_output_output.mp4" \
        --cond_aug 0. \
        --audio_path="/fsx/behavioural_computing_data/face_generation_data/AA_processed/${audio_dir_part}/${file_part}.pt" \
        --resize_size=512 \
        --use_latent=True \
        --num_steps=10 \
        --max_seconds=15 \
        --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
        --latent_folder=video_crop_emb \
        --video_folder=video_crop \
        --model_config=scripts/sampling/configs/svd_interpolation.yaml \
        --model_keyframes_config=scripts/sampling/configs/svd_keyframes_emo_cross.yaml \
        --get_landmarks=True \
        --landmark_folder=landmarks_crop \
        --overlap=${overlapping} \
        --chunk_size=5 \
        --audio_folder=audio \
        --audio_emb_folder=audio_emb \
        --output_folder=/data/home/antoni/results/${output_folder} \
        --keyframes_ckpt=${keyframes_ckpt} \
        --add_zero_flag=True \
        --emotion_folder=emotions \
        --extra_audio=False \

    echo "Processed $base_name"
done < "$file_list"
