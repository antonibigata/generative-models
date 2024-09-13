#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=00:00:00
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --error=/data/home/antoni/slurm_errors/generative_models/%j.err
#SBATCH --no-requeue
#SBATCH --account all
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/antoni/code/generative-models
srun python main.py --base configs/example_training/svd_dubbing_seq_cond.yaml --wandb True lightning.trainer.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/data/home/antoni/datasets/filelist_celebhq_text_aa_hdtf.txt \
    data.params.train.datapipeline.video_folder=video_crop  \
    data.params.train.datapipeline.audio_folder=audio \
    data.params.train.datapipeline.audio_emb_folder=audio_emb \
    data.params.train.datapipeline.latent_folder=video_crop_emb \
    data.params.train.loader.num_workers=6 \
    data.params.train.datapipeline.audio_in_video=False \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
    'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=2 \