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
from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio, save_audio_video

# from sgm.models.components.audio.Whisper import Whisper


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
            # audio_model = Whisper(model_size="large-v2", fps=25)
            # model.eval()
            # # Get audio embeddings
            # audio_embeddings = []
            # for chunk in torch.split(
            #     raw_audio, 750, dim=0
            # ):  # 750 is the max size of the audio chunks that can be processed by the model (= 30 seconds)
            #     audio_embeddings.append(audio_model(chunk.unsqueeze(0).cuda()))
            # audio = torch.cat(audio_embeddings, dim=1).squeeze(0)
    elif audio_path is not None and audio_path.endswith(".pt"):
        audio = torch.load(audio_path)
        raw_audio_path = audio_path.replace(".pt", ".wav").replace("_whisper_emb", "")

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, raw_audio


def sample(
    input_path: str = "",  # Can either be image file or folder with image files
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    resize_size: Optional[int] = None,
    version: str = "svd_image",
    fps_id: int = 25,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    autoregressive: int = 1,
    strength: float = 1.0,
    use_latent: bool = False,
    # degradation: int = 1,
    model_config: Optional[str] = None,
    max_seconds: Optional[int] = None,
    lora_path: Optional[str] = None,
    force_uc_zero_embeddings=[
        "cond_frames",
        "cond_frames_without_noise",
    ],
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "sd_2_1":
        num_frames = default(num_frames, 14)
        output_folder = default(output_folder, "outputs/simple_image_sample/sd_2_1/")
        model_config = "scripts/sampling/configs/sd_2_1.yaml"
        n_audio_frames = 2
    elif version == "svd_image":
        num_frames = default(num_frames, 14)
        output_folder = default(output_folder, "outputs/simple_image_sample/svd_image/")
        model_config = "scripts/sampling/configs/svd_image.yaml"
        n_audio_frames = 2

    if use_latent:
        input_key = "latents"
    else:
        input_key = "frames"

    model, filter, n_batch = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        input_key,
    )

    # model.en_and_decode_n_samples_a_time = decoding_t
    if lora_path is not None:
        model.init_from_ckpt(lora_path, remove_keys_from_weights=None)
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
    assert input_path == ""
    for input_img_path in all_img_paths:
        # if autoregressive > 1:
        #     raise ValueError(
        #         "Autoregressive sampling is not supported for videos. (Need to modify how to handle video)"
        #     )
        video = read_video(video_path, output_format="TCHW")[0]
        video = (video / 255.0) * 2.0 - 1.0

        video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
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

        # conditions_list, audio_list, emb_list = create_interpolation_inputs(model_input, audio, num_frames, video_emb)
        # if h % 64 != 0 or w % 64 != 0:
        #     width, height = map(lambda x: x - x % 64, (w, h))
        #     width = min(width, 1024)
        #     height = min(height, 576)
        #     video = torch.nn.functional.interpolate(video, (height, width), mode="bilinear")
        # video = model.encode_first_stage(video.cuda())
        cond = model_input[0]
        emb = video_emb[0] if video_emb is not None else None

        samples_list = []
        for i in tqdm(
            range(num_frames - 1, len(audio), num_frames), desc="Autoregressive", total=len(audio) // num_frames
        ):
            condition = cond
            audio_indexes = get_audio_indexes(i, n_audio_frames, len(audio))
            audio_cond = audio[audio_indexes].unsqueeze(0).cuda()
            # audio_cond = audio_list[i].unsqueeze(0).to(device)
            condition = condition.unsqueeze(0).to(device)
            embbedings = emb.unsqueeze(0).to(device) if emb is not None else None
            # print(condition.shape, embbedings.shape if embbedings is not None else None, audio_cond.shape)
            H, W = condition.shape[-2:]
            assert condition.shape[1] == 3
            F = 8
            C = 4
            shape = (1, C, H // F, W // F)
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
            value_dict["cond_frames_without_noise"] = condition
            if embbedings is not None:
                value_dict["cond_frames"] = embbedings + cond_aug * torch.randn_like(embbedings)
            else:
                value_dict["cond_frames"] = condition + cond_aug * torch.randn_like(condition)
            value_dict["cond_aug"] = cond_aug
            value_dict["audio_emb"] = audio_cond

            with torch.no_grad():
                with torch.autocast(device):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, 1],
                        T=1,
                        device=device,
                    )

                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=force_uc_zero_embeddings,
                    )

                    for k in ["crossattn"]:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=1)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=1)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=1)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=1)

                    video = torch.randn(shape, device=device)

                    additional_model_inputs = {}
                    # additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch, num_frames).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c):
                        return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                    samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
                    samples_x = model.decode_first_stage(samples_z)

                    # Replace first and last by condition
                    # samples_x[0] = condition.squeeze(0)[:, 0]
                    # samples_x[-1] = condition.squeeze(0)[:, 1]

                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                    # image = samples[-1] * 2.0 - 1.0
                    video = None

                    samples_list.append(samples)  # Keep last frame of last chunk
                    # samples_list.append(samples)

        samples_list.insert(0, torch.clamp((condition + 1.0) / 2.0, min=0.0, max=1.0))
        samples = torch.concatenate(samples_list)
        samples = torchvision.utils.make_grid(samples)

        os.makedirs(output_folder, exist_ok=True)
        base_count = len(glob(os.path.join(output_folder, "*.png")))
        video_path = os.path.join(output_folder, f"{base_count:06d}.png")
        samples = (rearrange(samples, "c h w -> h w c") * 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(samples).save(video_path)


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

    # if T is not None:
    batch["num_video_frames"] = 1

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    input_key: str,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config["model"]["params"]["input_key"] = input_key

    config.model.params.sampler_config.params.num_steps = num_steps
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

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter, n_batch


if __name__ == "__main__":
    Fire(sample)
