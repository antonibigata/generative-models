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
from scripts.util.vae_wrapper import VaeWrapper
from tqdm import tqdm


def default(value, default):
    return default if value is None else value


decord.bridge.set_bridge("torch")

"""

python scripts/util/video_to_latent.py --filelist /fsx/stellab/HDTF/filelist_videos_val.txt --diffusion_type video --resolution 512 --chunk_size 10

"""
def process_image(image, resolution=None):
    if resolution is not None:
        image = torch.nn.functional.interpolate(image.float(), size=resolution, mode="bilinear", align_corners=False)
    image = image / 127.5 - 1.0
    return image


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
    # parser.add_argument("--extra_name", type=str, default="")
    args = parser.parse_args()

    video_files = ['/fsx/rs2517/data/HDTF/cropped_videos_original/RD_Radio18_000.mp4']

    # for filelist in args.filelist:
    #     if filelist.endswith(".txt"):
    #         with open(filelist, "r") as f:
    #             files = f.readlines()
    #         video_files += [x.strip() for x in files]
    #     else:
    #         video_files.extend(list(Path(filelist).glob("*.mp4")))

    # Load the model
    model = VaeWrapper(args.diffusion_type)
    for name, param in model.named_parameters():
        print(name, param.shape)
    
    random.shuffle(video_files)
    missing = 0

    # model.disable_slicing()

    for video_file in tqdm(video_files, desc="Generating latent vectors"):
        
        # try:
            video_file = str(video_file)
            print('video_file', video_file)

            is_safe = "safetensors" if not args.save_as_tensor else "pt"
            
            out_file = Path(video_file).stem + f"_{args.diffusion_type}_{args.resolution}" + f"_latent.{is_safe}"
            out_path = Path(video_file).parent / out_file
            # out_path = Path(str(out_path).replace(args.in_dir, args.out_dir))            
            os.makedirs(str(out_path.parent), exist_ok=True)

            # if out_path.exists() and not args.force_recompute:
            #     continue

            if args.only_missing:
                missing += 1
                continue

            def encode_video(video):
                video = rearrange(video, "t h w c -> c t h w")
                print('video 2', video.shape)
                vid_rez = min(video.shape[-1], video.shape[-2])
                # to_rez = min(default(args.resolution, vid_rez), vid_rez)
                to_rez = default(args.resolution, vid_rez)
                video = process_image(video, to_rez)
                print('video 3', video.shape)
                encoded = rearrange(model.encode_video(video.cuda().unsqueeze(0)).squeeze(0), "c t h w -> t c h w")
                return encoded

            # Run the model
            with torch.no_grad():
                if args.chunk_size is not None:
                    encoded = []
                    video_reader = decord.VideoReader(video_file)
                    for i in tqdm(range(0, len(video_reader), args.chunk_size), leave=False, desc="Chunking"):
                        video = video_reader.get_batch(range(i, min(i + args.chunk_size, len(video_reader))))
                        print('video', video.shape)
                        encoded_chunk = encode_video(video)
                        print('encoded_chunk', encoded_chunk.shape)
                        quit()
                        encoded.append(encoded_chunk)
                    # Process the last chunk
                    if i < len(video_reader):
                        video = video_reader.get_batch(range(i, len(video_reader)))
                        encoded_chunk = encode_video(video)
                        encoded.append(encoded_chunk)
                    encoded = torch.cat(encoded, dim=0)
                else:
                    video_reader = decord.VideoReader(video_file)
                    video = video_reader.get_batch(range(len(video_reader)))
                    encoded = encode_video(video)

            # Create output path in same directory as video

            quit()
            # Save the latent vector
            if args.save_as_tensor:
                torch.save(encoded.cpu(), out_path)
            else:
                save_file({"latents": encoded.cpu(), "init_rez": torch.tensor(video.shape[-2:])}, out_path)
        # except:
        #     print(f"Failed for file {video_file}")

    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
