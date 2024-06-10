#!/bin/bash
#SBATCH --job-name=antoni_project
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=8
#SBATCH --time=00:00:00
#SBATCH --output=/data/home/antoni/slurm_logs/generative_models/%j.out
#SBATCH --error=/data/home/antoni/slurm_errors/generative_models/%j.err
#SBATCH --no-requeue
#SBATCH --exclude=a100-st-p4d24xlarge-90,a100-st-p4d24xlarge-67,a100-st-p4d24xlarge-35,a100-st-p4d24xlarge-51,a100-st-p4d24xlarge-55,a100-st-p4d24xlarge-70,a100-st-p4d24xlarge-71,a100-st-p4d24xlarge-38,a100-st-p4d24xlarge-83
source /data/home/antoni/miniconda3/etc/profile.d/conda.sh
conda activate svd
export WANDB_ENTITY=animator
export NCCL_SOCKET_IFNAME=ens32
export HYDRA_FULL_ERROR=1
export MODULEPATH=/opt/slurm/etc/files/modulesfiles:$MODULEPATH
export NCCL_TOPO_FILE=opt/aws-ofi-nccl/share/aws-ofi-nccl/xml/p4de-24xl-topo.xml
cd /data/home/antoni/code/generative-models

module load cuda/12.1 \
 nccl/2.18.3-cuda.12.1 \
 nccl_efa/1.24.1-nccl.2.18.3-cuda.12.1

srun python main.py --base configs/example_training/svd_interpolation_no_emb.yaml --wandb True lightning.trainer.num_nodes 8 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
    lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb \
    'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=2 \
    model.params.loss_fn_config.params.lambda_lower=2.