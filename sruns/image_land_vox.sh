#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=071:59:59
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --error=/data/home/antoni/slurm_errors/generative_models/%j.err
#SBATCH --no-requeue
#SBATCH --account all
#SBATCH --exclude=a100-st-p4d24xlarge-75
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/antoni/code/generative-models
srun python main.py --base configs/example_training/svd_image_land.yaml --wandb True lightning.trainer.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=1.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/voxceleb2_proper.txt \
    data.params.train.datapipeline.video_folder=/fsx/behavioural_computing_data/voxceleb2  \
    data.params.train.datapipeline.audio_folder=/fsx/rs2517/data/voxceleb2_wav2vec2_feats \
    data.params.train.datapipeline.latent_folder=/fsx/rs2517/data/voxceleb2_sd_latent \
    data.params.train.datapipeline.audio_in_video=True \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    data.params.train.loader.num_workers=4 \
    lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 data.params.train.datapipeline.virtual_increase=10 \
    model.params.network_config.params.audio_cond_method=to_time_emb_image data.params.train.loader.batch_size=32 \
    
