#!/bin/bash
#SBATCH --job-name=vox_box_29
#SBATCH --partition=learnai4rl  # learnai4rl learnai
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=00:00:00
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

# --logdir /fsx/stellab/checkpoints_diffusion_dubbing/ 
# 'model.params.remove_keys_from_weights=[model.diffusion_model]' -> remove pre-trained weights


# 'model.params.remove_keys_from_weights=[model.diffusion_model.label_emb.0.0.weight, model.diffusion_model.input_blocks.0.0.weight]' # For SD2.1 ?
# model.params.ckpt_path=checkpoints/v2-1_512-ema-pruned.safetensors

# srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --no_date True --resume ./logs/exp_00_hdtf  lightning.trainer.num_nodes 4 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
#     lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
#     data.params.train.loader.batch_size=1 \
#     'model.params.remove_keys_from_weights=[model.diffusion_model]'


srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --no_date True --logdir /fsx/stellab/checkpoints_diffusion_dubbing/ --name vox_pretrained_box_test_29 lightning.trainer.num_nodes 4 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
    model.params.network_config.params.audio_cond_method=to_time_emb \
    data.params.train.loader.batch_size=1 \
    data.params.train.datapipeline.filelist=./file_list_train_vox.txt \
    data.params.train.datapipeline.dataset_path="/fsx/behavioural_computing_data/voxceleb2/dev" \
    data.params.train.datapipeline.audio_emb_path="/fsx/rs2517/data/voxceleb2_wav2vec2_feats/dev" \
    data.params.train.datapipeline.latent_path="/fsx/rs2517/data/voxceleb2_sd_latent/dev" \
    data.params.train.datapipeline.landmarks_path="/fsx/behavioural_computing_data/voxceleb2/dev/Retina_landmark" \
    data.params.train.datapipeline.audio_in_video=True \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    data.params.train.datapipeline.what_mask=box



#  srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --no_date True --resume /fsx/stellab/checkpoints_diffusion_dubbing/hdtf_SD2 lightning.trainer.num_nodes 4 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     data.params.train.datapipeline.filelist=/fsx/rs2517/data/lists/HDTF/filelist_videos_train.txt \
#     lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
#     data.params.train.loader.batch_size=1 \
#     'model.params.remove_keys_from_weights=[model.diffusion_model]'

######### Run 2: #########
# srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --logdir /fsx/stellab/checkpoints_diffusion_dubbing/ --resume /fsx/stellab/checkpoints_diffusion_dubbing/voxceleb_half_big_pretrained_2 --no_date True  lightning.trainer.num_nodes 4 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=half \
#     data.params.train.loader.batch_size=1 \
#     data.params.train.datapipeline.filelist=./file_list_train_vox.txt \
#     data.params.train.datapipeline.dataset_path="/fsx/behavioural_computing_data/voxceleb2/dev" \
#     data.params.train.datapipeline.audio_emb_path="/fsx/rs2517/data/voxceleb2_wav2vec2_feats/dev" \
#     data.params.train.datapipeline.latent_path="/fsx/rs2517/data/voxceleb2_sd_latent/dev" \
#     data.params.train.datapipeline.audio_in_video=True \
#     data.params.train.datapipeline.get_masks=False \
#     data.params.train.datapipeline.load_all_possible_indexes=False 


######### Run 3: #########
# srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --resume ./logs/exp_00_voxceleb_half_big --no_date True lightning.trainer.num_nodes 4 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=half \
#     data.params.train.loader.batch_size=1 \
#     data.params.train.datapipeline.filelist=./file_list_train_vox.txt \
#     data.params.train.datapipeline.dataset_path="/fsx/behavioural_computing_data/voxceleb2/dev" \
#     data.params.train.datapipeline.audio_emb_path="/fsx/rs2517/data/voxceleb2_wav2vec2_feats/dev" \
#     data.params.train.datapipeline.latent_path="/fsx/rs2517/data/voxceleb2_sd_latent/dev" \
#     data.params.train.datapipeline.audio_in_video=True \
#     data.params.train.datapipeline.get_masks=False \
#     data.params.train.datapipeline.load_all_possible_indexes=False 'model.params.remove_keys_from_weights=[model.diffusion_model]'


    
#  srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb False --no_date True --name test lightning.trainer.num_nodes 1 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     lightning.trainer.devices=1 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=cheeks \
#     'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=2 \


# srun python main.py --base configs/example_training/svd_dubbing_half.yaml --wandb True --projectname dubbing --no_date True --name exp_00_voxceleb_pretrained lightning.trainer.num_nodes 4 \
#     lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
#     lightning.trainer.devices=8 lightning.trainer.accumulate_grad_batches=1 \
#     model.params.network_config.params.audio_cond_method=to_time_emb data.params.train.datapipeline.what_mask=box \
#     data.params.train.loader.batch_size=2 'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' \
#     data.params.train.datapipeline.filelist=./file_list_train_vox.txt \
#     data.params.train.datapipeline.dataset_path="/fsx/behavioural_computing_data/voxceleb2/dev" \
#     data.params.train.datapipeline.audio_emb_path="/fsx/rs2517/data/voxceleb2_wav2vec2_feats/dev" \
#     data.params.train.datapipeline.latent_path="/fsx/rs2517/data/voxceleb2_sd_latent/dev" \
#     data.params.train.datapipeline.audio_in_video=True \
#     data.params.train.datapipeline.get_masks=False \
#     data.params.train.datapipeline.load_all_possible_indexes=False model.params.calculate_landmarks=True
