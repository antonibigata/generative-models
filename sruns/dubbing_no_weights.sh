#!/bin/bash
#SBATCH --job-name=dub_dif
#SBATCH --partition=learnai
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=071:59:59
#SBATCH --error=/fsx/stellab/slurms_info/err/%j_%t_log.err 
#SBATCH --output=/fsx/stellab/slurms_info/out/%j_%t_log.out
#SBATCH --no-requeue
#SBATCH --account all
source /data/home/stellab/miniconda3/etc/profile.d/conda.sh

conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb False --projectname dubbing --no_date True --name exp_00_voxceleb lightning.trainer.num_nodes 1 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
    lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
    data.params.train.loader.batch_size=1 'model.params.remove_keys_from_weights=[model.diffusion_model]' \

