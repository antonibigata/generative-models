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
srun python main.py --base configs/example_training/svd_interpolation_no_emb.yaml --wandb True lightning.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=1.e-4 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
    lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1