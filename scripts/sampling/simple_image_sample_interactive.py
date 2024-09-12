import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor, Grayscale, Resize
from tqdm import tqdm
from torchvision.io import read_video, write_video
import torchaudio
from safetensors.torch import load_file as load_safetensors
import torchvision

# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio

from pytorch_lightning import seed_everything

from scripts.demo.streamlit_helpers import *

SAVE_PATH = "outputs/svd/image/"


VERSION2SPECS = {
    "SVD-Image": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image.yaml",
        "ckpt": "/data/home/antoni/code/generative-models/logs/2024-07-12T14-16-48_example_training-svd_image/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt",
        # "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
}


def get_audio_indexes(main_index, n_audio_frames, max_len):
    # Get indexes for audio from both sides of the main index
    audio_ids = []
    # get audio embs from both sides of the GT frame
    audio_ids += [0] * max(n_audio_frames - main_index, 0)
    for i in range(max(main_index - n_audio_frames, 0), min(main_index + n_audio_frames + 1, max_len)):
        # for i in range(frame_ids[0], min(frame_ids[0] + self.n_audio_motion_embs + 1, n_frames)):
        audio_ids += [i]
    audio_ids += [max_len - 1] * max(main_index + n_audio_frames - max_len + 1, 0)
    return audio_ids


def get_audio_embeddings(audio_path: str, audio_rate: int = 16000, fps: int = 25):
    # Process audio
    audio = None
    raw_audio = None
    if audio_path is not None and (audio_path.endswith(".wav") or audio_path.endswith(".mp3")):
        audio, sr = torchaudio.load(audio_path, channels_first=True)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=audio_rate)[0]
        samples_per_frame = math.ceil(audio_rate / fps)
        n_frames = audio.shape[-1] / samples_per_frame
        if not n_frames.is_integer():
            print("Audio shape before trim_pad_audio: ", audio.shape)
            audio = trim_pad_audio(audio, audio_rate, max_len_raw=math.ceil(n_frames) * samples_per_frame)
            print("Audio shape after trim_pad_audio: ", audio.shape)
        raw_audio = rearrange(audio, "(f s) -> f s", s=samples_per_frame)

        if "whisper" in audio_path.lower():
            raise NotImplementedError("Whisper audio embeddings are not yet supported.")

    elif audio_path is not None and audio_path.endswith(".pt"):
        audio = torch.load(audio_path)
        raw_audio_path = audio_path.replace(".pt", ".wav").replace("_whisper_emb", "")

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, raw_audio


def load_img(display=True, key=None, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)


def get_full_audio_and_video(
    input_path, video_path, audio_path, video_folder, latent_folder, max_seconds, use_latent, resize_size, fps_id
):
    path = Path(input_path)
    all_img_paths = []
    if input_path == "":
        all_img_paths = [""]
    elif path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    assert input_path == ""
    for input_img_path in all_img_paths:
        # if autoregressive > 1:
        #     raise ValueError(
        #         "Autoregressive sampling is not supported for videos. (Need to modify how to handle video)"
        #     )
        video = read_video(video_path, output_format="TCHW")[0]
        video = (video / 255.0) * 2.0 - 1.0
        video = torch.nn.functional.interpolate(video, (512, 512), mode="bilinear")

        video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
        if video_folder is not None and latent_folder is not None:
            video_embedding_path = video_embedding_path.replace(video_folder, latent_folder)
        video_emb = None
        if use_latent:
            video_emb = load_safetensors(video_embedding_path)["latents"]

        audio, raw_audio = get_audio_embeddings(audio_path, 16000, fps_id + 1)
        if max_seconds is not None:
            max_frames = max_seconds * fps_id
            if video.shape[0] > max_frames:
                video = video[:max_frames]
                audio = audio[:max_frames]
                video_emb = video_emb[:max_frames] if video_emb is not None else None
                raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
        audio = audio.cuda()

        h, w = video.shape[2:]
        # if degradation > 1:
        #     video = torch.nn.functional.interpolate(video, (h // degradation, w // degradation), mode="bilinear")
        #     video = torch.nn.functional.interpolate(video, (h, w), mode="bilinear")
        #     # Save video
        #     out_path = os.path.join(output_folder, "degraded.mp4")
        #     write_video(out_path, (((video.permute(0, 2, 3, 1) + 1) / 2) * 255.0).cpu(), fps_id)

        if input_img_path is None or input_img_path == "":
            model_input = video.cuda()
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                if resize_size is not None:
                    width, height = (resize_size, resize_size) if isinstance(resize_size, int) else resize_size
                else:
                    width = min(width, 1024)
                    height = min(height, 576)
                model_input = torch.nn.functional.interpolate(model_input, (height, width), mode="bilinear").squeeze(0)
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

        return model_input, video_emb, audio, raw_audio


def get_model_input(model_input, video_emb, audio, index):
    cond = model_input[0]
    emb = video_emb[0] if video_emb is not None else None
    condition = cond
    audio_indexes = get_audio_indexes(index, 2, len(audio))
    audio_cond = audio[audio_indexes].unsqueeze(0).cuda()
    # audio_cond = audio_list[i].unsqueeze(0).to(device)
    condition = condition.unsqueeze(0)
    embbedings = emb.unsqueeze(0) if emb is not None else None

    value_dict = {}
    # value_dict["motion_bucket_id"] = motion_bucket_id
    # value_dict["fps_id"] = fps_id
    # value_dict["cond_aug"] = cond_aug
    cond_aug = 0.0
    value_dict["cond_frames_without_noise"] = condition
    if embbedings is not None:
        value_dict["cond_frames"] = embbedings + cond_aug * torch.randn_like(embbedings)
    else:
        value_dict["cond_frames"] = condition + cond_aug * torch.randn_like(condition)
    value_dict["cond_aug"] = cond_aug
    value_dict["audio_emb"] = audio_cond
    return value_dict, model_input[index].unsqueeze(0)


def run_imgpred(
    state,
    version,
    version_dict,
    # model_input,
    # video_emb,
    # audio,
    filter=None,
    stage2strength=None,
):
    model_input = state["model_input"]
    video_emb = state["video_emb"]
    audio = state["audio"]
    H = st.number_input("H", value=version_dict["H"], min_value=64, max_value=2048)
    W = st.number_input("W", value=version_dict["W"], min_value=64, max_value=2048)
    C = version_dict["C"]
    F = version_dict["f"]

    zero_embeddings = st.multiselect(
        "Zero Embeddings",
        ["cond_frames_without_noise", "cond_frames", "audio_emb"],
        default=None,
    )

    index_to_predict = st.number_input("Index to Predict", value=0, min_value=0, max_value=1000)

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    # value_dict = init_embedder_options(
    #     get_unique_embedder_keys_from_conditioner(state["model"].conditioner), init_dict
    # )
    value_dict, gt = get_model_input(model_input, video_emb, audio, index_to_predict)
    sampler, num_rows, num_cols = init_sampling(stage2strength=stage2strength)
    num_samples = num_rows * num_cols
    value_dict["num_video_frames"] = 1

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        out = do_sample(
            state["model"],
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=zero_embeddings,
            return_latents=False,
            filter=filter,
            gt=gt,
            T=1,
        )
        return out


@st.cache(allow_output_mutation=True)
def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    input_key: str,
):
    state = dict()
    if "model" not in state:
        config = OmegaConf.load(config)
        if device == "cuda":
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device

        config["model"]["params"]["input_key"] = input_key

        # config.model.params.sampler_config.params.num_steps = num_steps
        if "num_frames" in config.model.params.sampler_config.params.guider_config.params:
            config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

        if "IdentityGuider" in config.model.params.sampler_config.params.guider_config.target:
            n_batch = 1
        else:
            n_batch = 2  # Conditional and unconditional
        if device == "cuda":
            # with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
        else:
            model = instantiate_from_config(config.model).to(device).eval()
        state["model"] = model
        # filter = DeepFloydDataFiltering(verbose=False, device=device)
    return state


if __name__ == "__main__":
    st.title("Image SVD")
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    # if st.checkbox("Load Model"):
    #     mode = "SVDImage"
    # else:
    #     mode = "skip"
    st.write("__________________________")

    # set_lowvram_mode(st.checkbox("Low vram mode", True))

    set_lowvram_mode(st.checkbox("Low vram mode", False))

    seed = st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9))
    seed_everything(seed)

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, version))

    audio_path = st.text_input("Audio Path", value="/fsx/rs2517/data/HDTF/audio/WDA_BarackObama_001_wav2vec2_emb.pt")
    video_path = st.text_input(
        "Video Path", value="/fsx/rs2517/data/HDTF/cropped_videos_original/WDA_BarackObama_001.mp4"
    )
    video_folder = st.text_input("Video Folder", value="")
    latent_folder = st.text_input("Latent Folder", value="")
    max_seconds = st.number_input("Max Seconds", value=10, min_value=1, max_value=60)

    # state = init_st(version_dict, load_filter=False, load_ckpt=False)
    state = load_model(
        version_dict["config"],
        "cuda",
        1,
        1,
        "latents",
    )

    # if state["msg"]:
    #     st.info(state["msg"])
    # model = state["model"]

    if "model_input" not in state:
        model_input, video_emb, audio, raw_audio = get_full_audio_and_video(
            "", video_path, audio_path, video_folder, latent_folder, max_seconds, True, 512, 24
        )
        state["model_input"] = model_input
        state["video_emb"] = video_emb
        state["audio"] = audio
        state["raw_audio"] = raw_audio

    stage2strength = None
    finish_denoising = False

    # if mode == "SVDImage":
    out = run_imgpred(
        state,
        version,
        version_dict,
        # model_input,
        # video_emb,
        # audio,
        filter=state.get("filter"),
        stage2strength=stage2strength,
    )
    # elif mode == "skip":
    #     out = None
    # else:
    #     raise ValueError(f"unknown mode {mode}")
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None

    if save_locally and samples is not None:
        perform_save_locally(save_path, samples)
