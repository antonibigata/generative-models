#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=72:00:00
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --error=/data/home/antoni/slurm_errors/generative_models/%j.err
#SBATCH --no-requeue
#SBATCH --account all
#SBATCH --exclude=a100-st-p4d24xlarge-2,a100-st-p4d24xlarge-61
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/antoni/code/generative-models
srun python main.py --resume logs/2024-09-10T10-05-28_example_training-svd_keyframes_emo_cross --base configs/example_training/svd_keyframes_emo_cross.yaml --wandb True lightning.trainer.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/data/home/antoni/datasets/filelist_video_aa.txt \
    data.params.train.datapipeline.video_folder=video_crop  \
    data.params.train.datapipeline.audio_folder=audio \
    data.params.train.datapipeline.audio_emb_folder=audio_emb \
    data.params.train.datapipeline.latent_folder=video_crop_emb \
    data.params.train.loader.num_workers=6 \
    data.params.train.datapipeline.audio_in_video=False \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=both_keyframes \
    data.params.train.loader.batch_size=2 \
    model.params.loss_fn_config.params.lambda_lower=2. data.params.train.datapipeline.virtual_increase=1 \
    data.params.train.datapipeline.select_randomly=False 'model.params.to_freeze=[]' 'model.params.to_unfreeze=[]' \
    data.params.train.datapipeline.balance_datasets=False data.params.train.datapipeline.need_cond=True data.params.train.datapipeline.get_separate_id=True  \
    model.params.network_wrapper.params.add_embeddings=True model.params.network_config.params.in_channels=12 'model.params.remove_keys_from_weights=["model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.label_emb.0.0.weight"]'