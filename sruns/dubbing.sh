#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
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
cd /data/home/rs2517/code/generative-models-dub
srun python main.py --resume logs/2024-05-14T14-34-10_example_training-svd_dubbing_half --base configs/example_training/svd_dubbing_half.yaml --wandb True lightning.trainer.num_nodes 4 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
    lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
    'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=2 \