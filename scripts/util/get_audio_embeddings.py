"""
Script to get audio embeddings for the audio files in the dataset.
"""

import argparse
import glob
import math
import os
import random
import sys
import gc

import torch
import torchaudio
import torchvision
from einops import rearrange
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.util.audio_wrapper import AudioWrapper
from sgm.util import trim_pad_audio


def make_into_multiple_of(x, multiple, dim=0):
    """
    Make the torch tensor into a multiple of the given number.
    """
    if x.shape[dim] % multiple != 0:
        x = torch.cat(
            [
                x,
                torch.zeros(
                    *x.shape[:dim],
                    multiple - (x.shape[dim] % multiple),
                    *x.shape[dim + 1 :],
                ).to(x.device),
            ],
            dim=dim,
        )
    return x


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
argparser.add_argument("--video_fps", type=int, default=25)
argparser.add_argument("--audio_rate", type=int, default=16000)
argparser.add_argument(
    "--audio_folder", type=str, default="", help="Name of audio folder following structure in README file"
)
argparser.add_argument(
    "--video_folder", type=str, default="", help="Name of video folder following structure in README file"
)
argparser.add_argument("--in_dir", type=str)
argparser.add_argument("--out_dir", type=str)
argparser.add_argument(
    "--recompute", action="store_true", help="Recompute audio embeddings even if they already exist"
)
argparser.add_argument("--skip_video", action="store_true", help="Skip video processing")
argparser.add_argument(
    "--count_missing", action="store_true", help="Only count the number of missing audio embeddings"
)
argparser.add_argument("--max_size", type=int, default=None, help="Maximum size of audio frames to process at once")
args = argparser.parse_args()


def calculate_new_frame_count(original_fps, target_fps, original_frame_count):
    # Calculate the duration of the original video
    duration = original_frame_count / original_fps

    # Calculate the new frame count
    new_frame_count = duration * target_fps

    # Round to the nearest integer
    return round(new_frame_count)


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return frame_count


@torch.no_grad()
def get_audio_embeddings(audio_path, output_path, model_size, fps, video_fps):
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
        audio_files = glob.glob(audio_path)
        # + glob.glob(os.path.join(audio_path, "**/*.wav"))
        # + glob.glob(os.path.join(audio_path, "**/**/*.wav"))

    audio_rate = args.audio_rate
    a2v_ratio = fps / float(audio_rate)
    samples_per_frame = math.ceil(1 / a2v_ratio)

    # Load model
    model = AudioWrapper(model_type=args.model_type, model_size=model_size, fps=fps)
    model.eval()
    model.cuda()

    # Shuffle audio files
    random.shuffle(audio_files)

    # Get audio embeddings
    missing_count = 0
    for audio_file in tqdm(audio_files, desc="Getting audio embeddings"):
        try:
            audio_file_name = os.path.basename(audio_file)
            audio_file_name = os.path.splitext(audio_file_name)[0]
            audio_file_name = audio_file_name + f"_{args.model_type}_emb.pt"
            audio_file_name = os.path.join(os.path.dirname(audio_file), audio_file_name)
            if os.path.exists(audio_file_name.replace(args.in_dir, args.out_dir)) and not args.recompute:
                continue

            missing_count += 1
            if args.count_missing:
                continue

            video_path = audio_file.replace(args.audio_folder, args.video_folder).replace(".wav", ".mp4")
            if "AA_processed" in video_path:
                video_path = video_path.replace(".mp4", "_output_output.mp4")
                video_fps = 60
            if not args.skip_video and not os.path.exists(video_path):
                print(f"Video file {video_path} does not exist. Skipping...")
                continue

            audio = None

            if not args.skip_video:
                # frames, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")
                # Free up memory
                len_video = get_video_duration(video_path)

                # sr = info.get("audio_fps", None)
                # len_video = frames.shape[0]
                max_len_sec = len_video / video_fps

                # del frames
                # gc.collect()
                # torch.cuda.empty_cache()
                if video_fps != fps:
                    len_video = calculate_new_frame_count(video_fps, fps, len_video)
            else:
                max_len_sec = None

            # print(audio.nelement() == 0, audio_file)
            # Load audio
            if audio is None or audio.nelement() == 0:
                audio, sr = torchaudio.load(audio_file)
            if sr != audio_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=audio_rate)[0]
            if audio.dim() == 2:
                audio = audio.mean(0, keepdim=True)

            if args.model_type == "wav2vec2":
                audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
                if max_len_sec is not None:
                    audio = trim_pad_audio(audio, audio_rate, max_len_sec=max_len_sec)[0]
                audio = make_into_multiple_of(audio, samples_per_frame, dim=0)
                audio_frames = rearrange(audio, "(f s) -> f s", s=samples_per_frame)
                if not args.skip_video:
                    assert audio_frames.shape[0] == len_video, f"{audio_frames.shape} != {len_video}"
                audio = rearrange(audio_frames, "f s -> () (f s)")
                # print(audio.shape, max_len_sec * audio_rate)
                audio_embeddings = model.encode_audio(audio.cuda())
                audio_embeddings = audio_embeddings.cpu()  # Move to CPU immediately
                del audio
                torch.cuda.empty_cache()
            else:
                audio = trim_pad_audio(audio, audio_rate, max_len_sec=max_len_sec)[0]
                audio = make_into_multiple_of(audio, samples_per_frame, dim=0)
                audio_frames = rearrange(audio, "(f s) -> f s", s=samples_per_frame)
                if not args.skip_video and audio_frames.shape[0] - len_video == 1:
                    audio_frames = audio_frames[:len_video]
                assert audio_frames.shape[0] == len_video, f"{audio_frames.shape} != {len_video}"
                if audio_frames.shape[0] % 2 != 0:
                    audio_frames = torch.cat([audio_frames, torch.zeros(1, samples_per_frame)], dim=0)

                # Get audio embeddings
                if args.max_size is not None and audio_frames.shape[0] > args.max_size:
                    # Split into 2 chunks if over max size
                    mid = audio_frames.shape[0] // 2
                    chunk1 = audio_frames[:mid].cuda()
                    chunk2 = audio_frames[mid:].cuda()

                    embeddings1 = model.encode_audio(chunk1)
                    embeddings2 = model.encode_audio(chunk2)

                    audio_embeddings = torch.cat([embeddings1.cpu(), embeddings2.cpu()], dim=0)

                else:
                    audio_embeddings = model.encode_audio(audio_frames.cuda())

                if not args.skip_video and audio_embeddings.shape[0] - len_video == 1:
                    audio_embeddings = audio_embeddings[:len_video]
                audio_embeddings = audio_embeddings.cpu()  # Move to CPU immediately
                del audio_frames
                torch.cuda.empty_cache()

            if not args.skip_video:
                assert audio_embeddings.shape[0] == len_video, f"{audio_embeddings.shape} != {len_video}"
            audio_file_name = audio_file_name.replace(args.in_dir, args.out_dir)
            os.makedirs(os.path.dirname(audio_file_name), exist_ok=True)
            torch.save(audio_embeddings.squeeze(0).cpu(), audio_file_name)
        except Exception as e:
            print(f"Failed for file {audio_file}: {str(e)}")
            # print(audio_frames.shape)
            continue  # Continue to the next file instead of raising the exception

    print("Count missing: ", missing_count)
    # Free up memory after processing all files
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    get_audio_embeddings(
        args.audio_path,
        args.output_path,
        args.model_size,
        args.fps,
        args.video_fps,
    )
