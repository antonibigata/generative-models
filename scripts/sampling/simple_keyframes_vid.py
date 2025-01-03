import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import random
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
import pytorch_lightning as pl

# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio, save_audio_video

# from sgm.models.components.audio.Whisper import Whisper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def create_prediction_inputs(
    video, audio, num_frames, video_emb=None, emotions=None, overlap=1, skip_frames=0, double_first=False
):
    """
    Create inputs for the model using video and audio data.

    Parameters:
        video (numpy.ndarray): The video frames array.
        audio (numpy.ndarray): The audio frames array.
        num_frames (int): Number of frames per chunk.
        video_emb (numpy.ndarray, optional): Video embeddings. Default is None.
        overlap (int, optional): Number of overlapping frames between chunks. Default is 1.
        skip_frames (int, optional): Number of frames to skip between each selected frame. Default is 0.

    Returns:
        tuple: (gt_chunks, cond, audio_chunks, cond_emb)
    """
    assert video.shape[0] == audio.shape[0], "Video and audio must have the same number of frames"

    audio_chunks = []
    cond_emb = None
    gt_chunks = []
    emotions_chunks = []
    # Adjust the step size for the loop to account for overlap and skipping frames
    step_size = (num_frames - overlap) * (skip_frames + 1)

    if double_first:
        num_frames -= 1

    indexes_generated = []
    for i in range(0, video.shape[0] - (num_frames - 1) * (skip_frames + 1), step_size):
        first = video[i]
        try:
            last = video[i + (num_frames - 1) * (skip_frames + 1)]
        except IndexError:
            break  # Last chunk is smaller than num_frames

        # Collect frames and audio samples with skipping
        video_index = [i + j * (skip_frames + 1) for j in range(num_frames)]
        if double_first:
            video_index = [i] + video_index
        print(video_index)
        indexes_generated.extend(video_index)
        gt_chunks.append(video[video_index, :])
        audio_chunks.append(audio[video_index, :])
        if emotions is not None:
            print((emotions[0][video_index], emotions[1][video_index], emotions[2][video_index]))
            emotions_chunks.append((emotions[0][video_index], emotions[1][video_index], emotions[2][video_index]))
            # emotions_chunks.append(
            #     (
            #         -torch.ones_like(emotions[0][video_index]),
            #         torch.ones_like(emotions[1][video_index]),
            #         emotions[2][video_index],
            #     )
            # )

    cond = video[0]
    if video_emb is not None:
        cond_emb = video_emb[0]

    return gt_chunks, cond, audio_chunks, cond_emb, emotions_chunks, indexes_generated


def get_audio_embeddings(audio_path: str, audio_rate: int = 16000, fps: int = 25, extra_audio: bool = False):
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
        raw_audio_path = audio_path.replace(".pt", ".wav").replace("_wav2vec2_emb", "")
        if extra_audio:
            extra_audio_emb = torch.load(audio_path.replace("_wav2vec2_emb", "_beats_emb"))

            print(
                f"Loaded extra audio embeddings from {audio_path.replace('_wav2vec2_emb', '_beats_emb')} {extra_audio_emb.shape}."
            )
            min_size = min(audio.shape[0], extra_audio_emb.shape[0])
            audio = torch.cat([audio[:min_size], extra_audio_emb[:min_size]], dim=-1)
            print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, raw_audio


def sample_with_scale(
    model,
    scale,
    audio_list,
    conditions,
    embs,
    emotions_chunks,
    device,
    force_uc_zero_embeddings,
    num_frames,
    n_batch,
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    cond_aug,
    raw_audio,
    output_folder,
    fps_id,
    motion_bucket_id,
    strength,
    gt_chunks,
    num_steps,
    double_first,
):
    if scale is not None:
        model.sampler.guider.set_scale(scale)
    if num_steps is not None:
        model.sampler.set_num_steps(num_steps)
    samples_list = []
    gt_list = []
    for i in tqdm(range(len(audio_list)), desc="Autoregressive", total=len(audio_list)):
        condition = conditions
        print(len(audio_list), type(audio_list[i]), len(audio_list[i]))
        audio_cond = audio_list[i].unsqueeze(0).to(device)
        condition = condition.unsqueeze(0).to(device)
        embbedings = embs.unsqueeze(0).to(device) if embs is not None else None
        H, W = condition.shape[-2:]
        assert condition.shape[1] == 3, f"Conditioning frame must have 3 channels, got {condition.shape}."
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
        value_dict["cond_frames_without_noise"] = condition
        if embbedings is not None:
            value_dict["cond_frames"] = embbedings + cond_aug * torch.randn_like(embbedings)
        else:
            value_dict["cond_frames"] = condition + cond_aug * torch.randn_like(condition)
        value_dict["cond_aug"] = cond_aug
        value_dict["audio_emb"] = audio_cond

        if len(emotions_chunks) > 0:
            value_dict["valence"] = emotions_chunks[i][0].unsqueeze(0).to(device)
            value_dict["arousal"] = emotions_chunks[i][1].unsqueeze(0).to(device)
            value_dict["emo_labels"] = emotions_chunks[i][2].unsqueeze(0).to(device)

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
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in ["crossattn"]:
                    if c[k].shape[1] != num_frames:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                video = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch, num_frames).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
                samples_x = model.decode_first_stage(samples_z)

                # Replace first and last by condition
                # samples_x[0] = condition.squeeze(0)[:, 0]
                # samples_x[-1] = condition.squeeze(0)[:, 1]

                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                if double_first:
                    samples = samples[1:]

                # image = samples[-1] * 2.0 - 1.0
                video = None

                # samples = embed_watermark(samples)
                # samples = filter(samples)
                # if i != len(audio_list) - 1:
                #     gt_list.append(torch.clamp((gt_chunks[i][:-1] + 1.0) / 2.0, min=0.0, max=1.0))
                #     samples_list.append(samples[:-1])  # Remove last frame to avoid overlap
                # else:
                #     samples_list.append(samples)  # Keep last frame of last chunk
                #     gt_list.append(torch.clamp((gt_chunks[i] + 1.0) / 2.0, min=0.0, max=1.0))
                samples_list.append(samples)
                # gt_list.append(torch.clamp((gt_chunks[i][:-1] + 1.0) / 2.0, min=0.0, max=1.0))

    samples = torch.concatenate(samples_list)
    # gt = torch.concatenate(gt_list)

    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}_keyframes_scale_{scale}_num_steps_{num_steps}.mp4")
    # video_path_gt = os.path.join(output_folder, f"{base_count:06d}_gt.mp4")
    # writer = cv2.VideoWriter(
    #     video_path,
    #     cv2.VideoWriter_fourcc(*"MP4V"),
    #     fps_id + 1,
    #     (samples.shape[-1], samples.shape[-2]),
    # )
    vid = (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
    # gt_vid = (rearrange(gt, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
    # for frame in vid:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     writer.write(frame)
    # writer.release()
    if raw_audio is not None:
        raw_audio = rearrange(raw_audio, "f s -> () (f s)")

    save_audio_video(
        vid,
        audio=raw_audio,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=video_path,
        keep_intermediate=False,
    )

    # save_audio_video(
    #     gt_vid,
    #     audio=raw_audio,
    #     frame_rate=fps_id + 1,
    #     sample_rate=16000,
    #     save_path=video_path_gt,
    #     keep_intermediate=False,
    # )

    print(f"Saved video to {video_path}")


def sample(
    input_path: str = "",  # Can either be image file or folder with image files
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    landmark_folder: Optional[str] = None,
    emotion_folder: Optional[str] = None,
    version: str = "svd",
    fps_id: int = 24,
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
    skip_frames: int = 12,
    image_leak: bool = False,
    ckpt_path: Optional[str] = None,
    recurse: bool = False,
    double_first: bool = False,
    extra_audio: bool = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_keyframes_sample/svd/")
        # model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_keyframes_sample/svd_xt/")
        # model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_keyframes_sample/svd_image_decoder/")
        # model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_keyframes_sample/svd_xt_image_decoder/")
        # model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    if use_latent:
        input_key = "latents"
    else:
        input_key = "frames"

    set_seed(seed)

    model, filter, n_batch = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        input_key,
        image_leak,
        ckpt_path,
    )

    model.en_and_decode_n_samples_a_time = decoding_t
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
        video = torch.nn.functional.interpolate(video, (512, 512), mode="bilinear")

        video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
        if video_folder is not None and latent_folder is not None:
            video_embedding_path = video_embedding_path.replace(video_folder, latent_folder)
        video_emb = None
        if use_latent:
            video_emb = load_safetensors(video_embedding_path)["latents"]

        audio, raw_audio = get_audio_embeddings(audio_path, 16000, fps_id + 1, extra_audio)
        if max_seconds is not None:
            max_frames = max_seconds * fps_id
            if video.shape[0] > max_frames:
                video = video[:max_frames]
                audio = audio[:max_frames]
                video_emb = video_emb[:max_frames] if video_emb is not None else None
                raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
        audio = audio.cuda()

        emotions = None
        if emotion_folder is not None:
            emotions_path = video_path.replace(video_folder, emotion_folder).replace(".mp4", ".pt")
            emotions = torch.load(emotions_path)
            emotions = (
                emotions["valence"][:max_frames],
                emotions["arousal"][:max_frames],
                emotions["labels"][:max_frames],
            )

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

        gt_chunks, conditions, audio_list, embs, emotions_chunks, indexes_generated = create_prediction_inputs(
            model_input,
            audio,
            num_frames,
            video_emb,
            emotions,
            skip_frames=skip_frames,
            double_first=double_first,
        )
        # if h % 64 != 0 or w % 64 != 0:
        #     width, height = map(lambda x: x - x % 64, (w, h))
        #     width = min(width, 1024)
        #     height = min(height, 576)
        #     video = torch.nn.functional.interpolate(video, (height, width), mode="bilinear")
        # video = model.encode_first_stage(video.cuda())
        if recurse:
            while True:
                scale = input("Enter the scale value (or 'stop' to end): ")
                if scale.lower() == "stop":
                    break
                try:
                    if "," not in scale:
                        scale = float(scale)
                    else:
                        scale = [float(x) for x in scale.split(",")]
                except ValueError:
                    print("Invalid input. Please enter a number or 'stop'.")
                    continue

                steps = input("Enter the number of steps (or press Enter to use default): ")
                if steps:
                    try:
                        steps = int(steps)
                    except ValueError:
                        print("Invalid input for steps. Using default.")
                        steps = None
                else:
                    steps = None

                sample_with_scale(
                    model,
                    scale,
                    audio_list,
                    conditions,
                    embs,
                    emotions_chunks,
                    device,
                    force_uc_zero_embeddings,
                    num_frames,
                    n_batch,
                    get_batch,
                    get_unique_embedder_keys_from_conditioner,
                    cond_aug,
                    raw_audio,
                    output_folder,
                    fps_id,
                    motion_bucket_id,
                    strength,
                    gt_chunks,
                    steps,
                    double_first,
                )
        else:
            sample_with_scale(
                model,
                None,
                audio_list,
                conditions,
                embs,
                emotions_chunks,
                device,
                force_uc_zero_embeddings,
                num_frames,
                n_batch,
                get_batch,
                get_unique_embedder_keys_from_conditioner,
                cond_aug,
                raw_audio,
                output_folder,
                fps_id,
                motion_bucket_id,
                strength,
                gt_chunks,
                None,
                double_first,
            )


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
    input_key: str,
    image_leak: bool = False,
    ckpt_path: Optional[str] = None,
):
    config = OmegaConf.load(config)
    if ckpt_path is not None:
        config.model.params.ckpt_path = ckpt_path
    # if device == "cuda":
    #     config.model.params.conditioner_config.params.emb_models[
    #         0
    #     ].params.open_clip_embedding_config.params.init_device = device

    config["model"]["params"]["input_key"] = input_key

    if "network_wrapper" in config.model.params:
        config["model"]["params"]["network_wrapper"]["params"]["fix_image_leak"] = image_leak

    config.model.params.sampler_config.params.num_steps = num_steps
    if "num_frames" in config.model.params.sampler_config.params.guider_config.params:
        config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    if "IdentityGuider" in config.model.params.sampler_config.params.guider_config.target:
        n_batch = 1
    elif "MultipleCondVanilla" in config.model.params.sampler_config.params.guider_config.target:
        n_batch = 3
    elif "AudioRefMultiCondGuider" in config.model.params.sampler_config.params.guider_config.target:
        n_batch = 3
    else:
        n_batch = 2  # Conditional and unconditional
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    # import thunder

    # model = thunder.jit(model)

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter, n_batch


if __name__ == "__main__":
    Fire(sample)
