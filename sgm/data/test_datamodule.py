from omegaconf import OmegaConf
from tqdm import tqdm

from sgm.data.video_image_datamodule import VideoDataModule


def test_video_data_module(config):
    module = VideoDataModule(config)

    module.setup("fit")

    train_loader = module.train_dataloader()

    # Test full train loader
    for batch in tqdm(train_loader, desc="Train loader", total=len(train_loader)):
        print("Cond frame shape: ", batch["cond_frames"].shape)
        print("Cond clean frame shape: ", batch["cond_frames_without_noise"].shape)
        print("Target frame shape: ", batch["latents"].shape)
        print("Audio shape: ", batch["audio_emb"].shape)


if __name__ == "__main__":
    train_config = OmegaConf.create(
        {
            "datapipeline": {
                "filelist": "/fsx/rs2517/data/lists/voxceleb2_proper.txt",
                "resize_size": 512,
                "audio_folder": "/fsx/rs2517/data/voxceleb2_wav2vec2_feats",
                "video_folder": "/fsx/behavioural_computing_data/voxceleb2",
                "video_extension": ".mp4",
                "audio_extension": ".wav",
                "latent_folder": "/fsx/rs2517/data/voxceleb2_sd_latent",
                "audio_in_video": True,
                "audio_rate": 16000,
                "num_frames": 14,
                "use_latent": True,
                "latent_type": "video",
                "latent_scale": 1,  # For backwards compatibility
                "from_audio_embedding": True,
                "load_all_possible_indexes": False,
                "allow_all_possible_permutations": False,
                "audio_emb_type": "wav2vec2",
                # cond_noise: [-3.0, 0.5]
                "cond_noise": 0.0,
                "motion_id": 60,
                # data_mean: null
                # data_std: null
                "additional_audio_frames": 2,
                "virtual_increase": 10,
                "use_latent_condition": True,
            },
            "loader": {
                "batch_size": 32,
                "num_workers": 4,
                "drop_last": True,
                "pin_memory": True,
                "persistent_workers": True,
            },
        }
    )

    test_video_data_module(train_config)
