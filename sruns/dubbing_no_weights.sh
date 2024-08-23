#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=071:59:59
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --no-requeue
#SBATCH --account all
<<<<<<< HEAD
=======
#SBATCH --exclude=a100-st-p4d24xlarge-46
>>>>>>> a7c72244eb98547df261bd0ed5fa6c63cabc80be
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /data/home/antoni/code/generative-models-dub
srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True lightning.trainer.num_nodes 4 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
    lightning.trainer.devices=4 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
<<<<<<< HEAD
    data.params.train.loader.batch_size=2 'model.params.remove_keys_from_weights=[model.diffusion_model]' \
=======
    data.params.train.loader.batch_size=1 'model.params.remove_keys_from_weights=[model.diffusion_model]' \
>>>>>>> a7c72244eb98547df261bd0ed5fa6c63cabc80be
