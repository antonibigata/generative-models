"""
Script to generate latent vectors from a video file.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random

import decord

# from torchvision.io import read_video
from einops import rearrange
from safetensors.torch import save_file
from tqdm import tqdm
import numpy as np

from sgm.modules.lipreader.lightnining import ModelModule
from sgm.modules.lipreader.preparation.detectors.retinaface.video_process import VideoProcess
from sgm.modules.lipreader.datamodule.transforms import VideoTransform
from sgm.data.data_utils import scale_landmarks


def default(value, default):
    return default if value is None else value


decord.bridge.set_bridge("torch")


def process_image(image, resolution=None):
    if resolution is not None:
        image = torch.nn.functional.interpolate(image.float(), size=resolution, mode="bilinear", align_corners=False)
    # image = image / 127.5 - 1.0
    return image


def get_lightning_module(ckpt_path):
    modelmodule = ModelModule()
    modelmodule.model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    return modelmodule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True, nargs="+")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--diffusion_type", type=str, default="stable")
    parser.add_argument("--if_oom", action="store_true", default=False)
    parser.add_argument("--save_as_tensor", action="store_true", default=False)
    parser.add_argument("--only_missing", action="store_true", default=False)
    parser.add_argument("--force_recompute", action="store_true", default=False)
    parser.add_argument("--to_extract", type=str, default="conformer")
    # parser.add_argument("--extra_name", type=str, default="")
    args = parser.parse_args()

    video_files = []

    for filelist in args.filelist:
        if filelist.endswith(".txt"):
            with open(filelist, "r") as f:
                files = f.readlines()
            video_files += [x.strip() for x in files]
        else:
            video_files.extend(list(Path(filelist).glob("*.mp4")))

    # Load the model
    ckpt_path = "/vol/paramonos2/projects/antoni/code/Personal/lipreader/models/vsr_trlrs3_base.max400.pth"
    model = get_lightning_module(ckpt_path)
    model.eval()

    random.shuffle(video_files)
    missing = 0

    video_transform = VideoTransform(subset="val", max_noise_level=250)
    video_process = VideoProcess(convert_gray=False)

    # model.disable_slicing()

    for video_file in tqdm(video_files, desc="Generating latent vectors"):
        try:
            video_file = str(video_file)

            is_safe = "safetensors" if not args.save_as_tensor else "pt"

            out_file = Path(video_file).stem + f"_{args.diffusion_type}_{args.resolution}" + f"_latent.{is_safe}"
            out_path = Path(video_file).parent / out_file
            out_path = Path(str(out_path).replace(args.in_dir, args.out_dir))
            os.makedirs(str(out_path.parent), exist_ok=True)

            if out_path.exists() and not args.force_recompute:
                continue

            if args.only_missing:
                missing += 1
                continue

            def encode_video(video, landmarks):
                video = rearrange(video, "t h w c -> t c h w")
                vid_rez = min(video.shape[-1], video.shape[-2])
                # to_rez = min(default(args.resolution, vid_rez), vid_rez)
                to_rez = default(args.resolution, vid_rez)
                video = process_image(video, to_rez)
                video = rearrange(video, "t c h w -> t h w c").numpy()

                landmarks = scale_landmarks(landmarks, (vid_rez, vid_rez), (to_rez, to_rez))
                video_proccessed = video_process(video, landmarks, True)
                video_proccessed = torch.tensor(video_proccessed)
                video_proccessed = video_proccessed.permute((0, 3, 1, 2))
                video_proccessed, t = video_transform(video_proccessed)

                encoded = model(video_proccessed.cuda(), extract_position=args.to_extract).squeeze(0)
                return encoded

            # Run the model
            with torch.no_grad():
                if args.chunk_size is not None:
                    encoded = []
                    video_reader = decord.VideoReader(video_file)
                    landmarks = np.load(video_file.replace(".mp4", ".npy"))
                    for i in tqdm(range(0, len(video_reader), args.chunk_size), leave=False, desc="Chunking"):
                        video = video_reader.get_batch(range(i, min(i + args.chunk_size, len(video_reader))))
                        encoded_chunk = encode_video(video, landmarks)
                        encoded.append(encoded_chunk)
                    # Process the last chunk
                    if i < len(video_reader):
                        video = video_reader.get_batch(range(i, len(video_reader)))
                        encoded_chunk = encode_video(video, landmarks)
                        encoded.append(encoded_chunk)
                    encoded = torch.cat(encoded, dim=0)
                else:
                    video_reader = decord.VideoReader(video_file)
                    landmarks = np.load(video_file.replace(".mp4", ".npy"))
                    video = video_reader.get_batch(range(len(video_reader)))
                    encoded = encode_video(video, landmarks)

            # Create output path in same directory as video

            # Save the latent vector
            if args.save_as_tensor:
                torch.save(encoded.cpu(), out_path)
            else:
                save_file({"latents": encoded.cpu(), "init_rez": torch.tensor(video.shape[-2:])}, out_path)
        except:
            print(f"Failed for file {video_file}")

    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
