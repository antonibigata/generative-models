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
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torchvision.io import read_video, write_video
import torchaudio

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config

# from sgm.models.components.audio.Whisper import Whisper


def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    autoregressive: int = 1,
    strength: float = 1.0,
    degradation: int = 1,
    audio_rate: int = 16000,
    model_config: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml" if model_config is None else model_config
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml" if model_config is None else model_config
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_image_decoder/")
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml" if model_config is None else model_config
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/")
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml" if model_config is None else model_config
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    model.en_and_decode_n_samples_a_time = decoding_t
    torch.manual_seed(seed)

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

    # audio = None
    # raw_audio = None
    # if audio_path is not None and (audio_path.endswith(".wav") or audio_path.endswith(".mp3")):
    #     audio, sr = torchaudio.load(audio_path, channels_first=True)
    #     if audio.shape[0] > 1:
    #         audio = audio.mean(0, keepdim=True)
    #     audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=audio_rate)[0]
    #     samples_per_frame = math.ceil(audio_rate / (fps_id + 1))
    #     n_frames = audio.shape[-1] / samples_per_frame
    #     if not n_frames.is_integer():
    #         print("Audio shape before trim_pad_audio: ", audio.shape)
    #         audio = trim_pad_audio(audio, audio_rate, max_len_raw=math.ceil(n_frames) * samples_per_frame)
    #         print("Audio shape after trim_pad_audio: ", audio.shape)
    #     audio = rearrange(audio, "(f s) -> f s", s=samples_per_frame)
    #     raw_audio = audio.clone()
    #     if "embeddings" in cfg.model.net.audio_encoder._target_:
    #         audio_model = Whisper(model_size="large-v3", fps=25)
    #         model.eval()
    #         # Get audio embeddings
    #         audio_embeddings = []
    #         for chunk in torch.split(
    #             audio, 750, dim=0
    #         ):  # 750 is the max size of the audio chunks that can be processed by the model (= 30 seconds)
    #             audio_embeddings.append(audio_model(chunk.unsqueeze(0).cuda()))
    #         audio = torch.cat(audio_embeddings, dim=1).squeeze(0)
    # elif audio_path is not None and audio_path.endswith(".pt"):
    audio_emb = torch.load(audio_path).to(device)
    audio_emb = repeat(audio_emb, "f c s -> (t f) c s", t=autoregressive).unsqueeze(0)
    audio_list = [audio_emb[:, i : i + num_frames] for i in range(0, audio_emb.shape[1], num_frames)]

    for input_img_path in all_img_paths:
        image = None
        if video_path is not None:
            if autoregressive > 1:
                raise ValueError(
                    "Autoregressive sampling is not supported for videos. (Need to modify how to handle video)"
                )
            video = read_video(video_path)[0]
            video = (video.permute(0, 3, 1, 2)[:num_frames] / 255.0) * 2.0 - 1.0
            h, w = video.shape[2:]
            if degradation > 1:
                video = torch.nn.functional.interpolate(video, (h // degradation, w // degradation), mode="bilinear")
                video = torch.nn.functional.interpolate(video, (h, w), mode="bilinear")
                # Save video
                out_path = os.path.join(output_folder, "degraded.mp4")
                write_video(out_path, (((video.permute(0, 2, 3, 1) + 1) / 2) * 255.0).cpu(), fps_id)

            if input_img_path is None or input_img_path == "":
                image = video[0].clone().cuda()
                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    width = min(width, 1024)
                    height = min(height, 576)
                    image = torch.nn.functional.interpolate(
                        image.unsqueeze(0), (height, width), mode="bilinear"
                    ).squeeze(0)
                    print(
                        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                width = min(width, 1024)
                height = min(height, 576)
                video = torch.nn.functional.interpolate(video, (height, width), mode="bilinear")
            video = model.encode_first_stage(video.cuda())

        else:
            video = None

        if image is None:
            with Image.open(input_img_path) as image:
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                w, h = image.size

                image = image.resize((576, 576))
                print(image.size)

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    image = image.resize((width, height))
                    print(
                        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )

                image = ToTensor()(image)
                image = image * 2.0 - 1.0

        or_image = image.clone().unsqueeze(0).to(device)

        samples_list = []
        for iter_idx in tqdm(range(autoregressive), desc="Autoregressive", total=autoregressive):
            image = image.unsqueeze(0).to(device)
            H, W = image.shape[2:]
            assert image.shape[1] == 3
            F = 8
            C = 4
            shape = (num_frames, C, H // F, W // F)
            if (H, W) != (576, 1024):
                print(
                    "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
                )
            if motion_bucket_id > 255:
                print("WARNING: High motion bucket! This may lead to suboptimal performance.")

            if fps_id < 5:
                print("WARNING: Small fps value! This may lead to suboptimal performance.")

            if fps_id > 30:
                print("WARNING: Large fps value! This may lead to suboptimal performance.")

            value_dict = {}
            value_dict["motion_bucket_id"] = motion_bucket_id
            value_dict["fps_id"] = fps_id
            value_dict["cond_aug"] = cond_aug
            value_dict["cond_frames_without_noise"] = or_image
            value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
            value_dict["cond_aug"] = cond_aug
            value_dict["audio_emb"] = audio_list[iter_idx]

            with torch.no_grad():
                with torch.autocast(device):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=[
                            "cond_frames",
                            "cond_frames_without_noise",
                        ],
                    )

                    for k in ["crossattn", "concat"]:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                    if video is None:
                        video = torch.randn(shape, device=device)

                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c):
                        return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                    samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
                    samples_x = model.decode_first_stage(samples_z)

                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                    image = samples[-1] * 2.0 - 1.0
                    video = None

                    # samples = embed_watermark(samples)
                    samples = filter(samples)
                    samples_list.append(samples)

        samples = torch.concatenate(samples_list)

        os.makedirs(output_folder, exist_ok=True)
        base_count = len(glob(os.path.join(output_folder, "*.mp4")))
        video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps_id + 1,
            (samples.shape[-1], samples.shape[-2]),
        )
        vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
        for frame in vid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(sample)
