#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:7                 # Request 8 GPUs on a single node
#SBATCH --ntasks=1                   # Run one task (the loop) per node
#SBATCH --cpus-per-task=12           # CPUs available for the loop
#SBATCH --nodes=1                    # Use a single node
#SBATCH --time=72:00:00              # Maximum runtime
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --error=/data/home/antoni/slurm_errors/generative_models/%j.err
#SBATCH --no-requeue
#SBATCH --account all
#SBATCH --nodelist=a100-st-p4d24xlarge-44
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
cd /data/home/antoni/code/generative-models
export PYTHONPATH=$PYTHONPATH:/data/home/antoni/code/generative-models

# Number of processes to start (match with number of GPUs)
NUM_PROCESSES=8

# Loop over available GPUs
for i in $(seq 0 $((NUM_PROCESSES-1))); do
    AUDIO_LIST="/data/home/antoni/datasets/splits_lrs3/16_splits/train/input_list2_${i}.txt"
    VIDEO_LIST="/data/home/antoni/datasets/splits_lrs3/16_splits/train/input_list1_${i}.txt"
    
    # Start a process for each GPU in the background
    CUDA_VISIBLE_DEVICES=$i python scripts/sampling/full_pipeline_emo.py \
        --filelist=${VIDEO_LIST} \
        --filelist_audio=${AUDIO_LIST} \
        --decoding_t 1 \
        --cond_aug 0. \
        --resize_size=512 \
        --use_latent=True \
        --max_seconds=14 \
        --force_uc_zero_embeddings='[cond_frames, audio_emb]' \
        --latent_folder=video_crop_emb \
        --video_folder=video_crop \
        --model_config=scripts/sampling/configs/svd_interpolation_high_quali.yaml \
        --model_keyframes_config=scripts/sampling/configs/svd_keyframes_emo_cross.yaml \
        --get_landmarks=False \
        --landmark_folder=landmarks_crop \
        --overlap=1 \
        --chunk_size=2 \
        --audio_folder=audio \
        --audio_emb_folder=audio_emb \
        --output_folder=/fsx/behavioural_computing_data/LRS3/keyface_videos \
        --keyframes_ckpt=logs/2024-11-09T17-26-06_example_training-keyframes_no_beats/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
        --interpolation_ckpt=/home/dinovgk/projects/generative-models/logs/2024-10-04T14-17-11_example_training-svd_interpolation_cross/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
        --double_first=False \
        --add_zero_flag=True \
        --emotion_folder=None \
        --extra_audio=both \
        --compute_until=45 \
        --audio_emb_type=wavlm --emotion_states='[neutral]' --recompute=True &
done

# Wait for all background processes to complete
wait