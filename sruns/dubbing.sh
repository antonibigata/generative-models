#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=71:00:00
#SBATCH --error=/fsx/stellab/slurms_info/err/%j_%t_log.err 
#SBATCH --output=/fsx/stellab/slurms_info/out/%j_%t_log.out
#SBATCH --no-requeue
#SBATCH --account all
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/antoni/code/generative-models
# srun python main.py --resume logs/2024-08-28T16-39-19_example_training-svd_dubbing_half --base configs/example_training/svd_dubbing_half.yaml --wandb True lightning.trainer.num_nodes 8 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     data.params.train.datapipeline.filelist=/data/home/antoni/datasets/filelist_celebhq_text_aa_hdtf.txt \
#     data.params.train.datapipeline.video_folder=video_crop  \
#     data.params.train.datapipeline.audio_folder=audio \
#     data.params.train.datapipeline.audio_emb_folder=audio_emb \
#     data.params.train.datapipeline.latent_folder=video_crop_emb \
#     data.params.train.loader.num_workers=6 \
#     data.params.train.datapipeline.audio_in_video=False \
#     data.params.train.datapipeline.load_all_possible_indexes=False \
#     lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
#     'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=2 \

srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --no_date True --logdir /fsx/stellab/checkpoints_diffusion_dubbing/ --name test_training lightning.trainer.num_nodes 8 \
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
    data.params.train.datapipeline.use_different_cond_frames=True 