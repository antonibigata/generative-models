"""
Script to get audio embeddings for the audio files in the dataset.
"""

import argparse
import os
import sys
import torchaudio
import torchvision
import torch
from einops import rearrange
import math
import glob
import random

from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sgm.util import trim_pad_audio
from scripts.util.audio_wrapper import AudioWrapper

argparser = argparse.ArgumentParser()
argparser.add_argument("--audio_path", type=str, default="data/audio_files.txt")
argparser.add_argument("--model_type", type=str, default="whisper", help="Model type: whisper or wavlm")
argparser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Path to save the embeddings, if None save in same directory as audio file",
)
argparser.add_argument(
    "--model_size", type=str, default="base", help="Model size: base, small, medium, large or large-v2"
)
argparser.add_argument("--fps", type=int, default=25)
argparser.add_argument("--audio_rate", type=int, default=16000)
argparser.add_argument(
    "--audio_folder", type=str, default="", help="Name of audio folder following structure in README file"
)
argparser.add_argument(
    "--video_folder", type=str, default="", help="Name of video folder following structure in README file"
)
argparser.add_argument(
    "--recompute", action="store_true", help="Recompute audio embeddings even if they already exist"
)
args = argparser.parse_args()


def get_audio_embeddings(audio_path, output_path, model_size, fps):
    """
    Get audio embeddings for the audio files in the dataset.
    """
    # Load audio files
    audio_files = []
    if audio_path.endswith(".txt"):
        with open(audio_path, "r") as f:
            for line in f:
                audio_files.append(line.strip())
    else:
        audio_files = (
            glob.glob(os.path.join(audio_path, "*.wav"))
            + glob.glob(os.path.join(audio_path, "**/*.wav"))
            + glob.glob(os.path.join(audio_path, "**/**/*.wav"))
        )

    audio_rate = args.audio_rate
    a2v_ratio = fps / float(audio_rate)
    samples_per_frame = math.ceil(1 / a2v_ratio)

    # Load model
    # model = Whisper(model_size=model_size, fps=fps)
    model = AudioWrapper(model_type=args.model_type, model_size=model_size, fps=fps)
    model.eval()
    model.cuda()

    # Shuffle audio files
    random.shuffle(audio_files)

    # Get audio embeddings
    for audio_file in tqdm(audio_files, desc="Getting audio embeddings"):
        audio_file_name = os.path.basename(audio_file)
        audio_file_name = os.path.splitext(audio_file_name)[0]
        audio_file_name = audio_file_name + f"_{args.model_type}_emb.pt"
        audio_file_name = os.path.join(
            output_path if output_path is not None else os.path.dirname(audio_file), audio_file_name
        )
        if os.path.exists(audio_file_name) and not args.recompute:
            continue

        video_path = audio_file.replace(args.audio_folder, args.video_folder).replace(".wav", ".mp4")
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist. Skipping...")
            continue
        frames, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")
        sr = info.get(["audio_fps"], None)
        max_len_sec = frames.shape[0] / fps

        # Load audio
        if audio is None:
            audio, sr = torchaudio.load(audio_file)
        if sr != audio_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=audio_rate)[0]
        audio = audio.mean(0, keepdim=True)
        # audio = make_into_multiple_of(audio, samples_per_frame, dim=0)
        audio = trim_pad_audio(audio, audio_rate, max_len_sec=max_len_sec)[0]
        audio_frames = rearrange(audio, "(f s) -> f s", s=samples_per_frame)
        assert audio_frames.shape[0] == frames.shape[0], f"{audio_frames.shape} != {frames.shape}"
        # if audio_frames.shape[0] % 2 != 0:
        #     audio_frames = torch.cat([audio_frames, torch.zeros(1, samples_per_frame)], dim=0)

        # Get audio embeddings
        audio_embeddings = model.encode_audio(audio_frames.cuda())

        torch.save(audio_embeddings.squeeze(0).cpu(), audio_file_name)


if __name__ == "__main__":
    get_audio_embeddings(
        args.audio_path,
        args.output_path,
        args.model_size,
        args.fps,
    )
