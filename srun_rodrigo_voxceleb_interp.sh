#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=071:59:59
#SBATCH --output=/data/home/rs2517/antonigo/out/%j.out
#SBATCH --no-requeue
#SBATCH --account all
source /data/home/rs2517/miniconda3/etc/profile.d/conda.sh
conda activate /fsx/rs2517/conda_envs/svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/rs2517/code/generative-models
srun python main.py --base configs/example_training/svd_interpolation_no_emb.yaml --wandb True lightning.trainer.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/voxceleb2_proper.txt \
    data.params.train.datapipeline.video_folder=/fsx/behavioural_computing_data/voxceleb2  \
    data.params.train.datapipeline.audio_folder=/fsx/rs2517/data/voxceleb2_whisper_feats \
    data.params.train.datapipeline.audio_emb_type=whisper \
    data.params.train.datapipeline.latent_folder=/fsx/rs2517/data/voxceleb2_sd_latent \
    data.params.train.datapipeline.audio_in_video=True \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=4 \
    model.params.network_config.params.audio_dim=1280 