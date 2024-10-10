import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io import read_video
import torchaudio
from safetensors.torch import load_file as load_safetensors

from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio, save_audio_video
from sgm.data.data_utils import (
    create_masks_from_landmarks_full_size,
    create_face_mask_from_landmarks,
    create_masks_from_landmarks_box,
    draw_kps_image,
    scale_landmarks,
)
# from sgm.models.components.audio.Whisper import Whisper


def load_landmarks(landmarks, original_size, index, target_size=(64, 64), is_dub=False, what_mask="box"):
    if is_dub:
        landmarks = landmarks[index, :][None, ...]
        if what_mask == "full":
            mask = create_masks_from_landmarks_full_size(landmarks, original_size[0], original_size[1], offset=-0.01)
        elif what_mask == "box":
            mask = create_masks_from_landmarks_box(landmarks, (original_size[0], original_size[1]), box_expand=0.0)
        else:
            mask = create_face_mask_from_landmarks(landmarks, original_size[0], original_size[1], mask_expand=0.05)
        # mask = create_masks_from_landmarks_full_size(landmarks, original_size[0], original_size[1], offset=-0.01)
        # Interpolate the mask to the target size
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode="nearest")
        return mask
    else:
        landmarks = landmarks[index]
        land_image = draw_kps_image(target_size, original_size, landmarks, rgb=True, pts_width=1)
        return torch.from_numpy(land_image).float() / 255.0


def merge_overlapping_segments(segments, overlap):
    """
    Merges overlapping segments by averaging overlapping frames.
    Segments have shape (b, t, ...), where 'b' is the number of segments,
    't' is frames per segment, and '...' are other dimensions.

    :param segments: Tensor of shape (b, t, ...)
    :param overlap: Integer, number of frames that overlap between consecutive segments.
    :return: Tensor of the merged video.
    """
    # Get the shape details
    b, t, *other_dims = segments.shape
    num_frames = (b - 1) * (t - overlap) + t  # Calculate the total number of frames in the merged video

    # Initialize the output tensor and a count tensor to keep track of contributions for averaging
    output_shape = [num_frames] + other_dims
    output = torch.zeros(output_shape, dtype=segments.dtype, device=segments.device)
    count = torch.zeros(output_shape, dtype=torch.float32, device=segments.device)

    current_index = 0
    for i in range(b):
        end_index = current_index + t
        # Add the segment to the output tensor
        output[current_index:end_index] += rearrange(segments[i], "... -> ...")
        # Increment the count tensor for each frame that's added
        count[current_index:end_index] += 1
        # Update the starting index for the next segment
        current_index += t - overlap

    # Avoid division by zero
    count[count == 0] = 1
    # Average the frames where there's overlap
    output /= count

    return output


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


def create_pipeline_inputs(
    video, audio, audio_interpolation, num_frames, video_emb, emotions=None, overlap=1, add_zero_flag=False, double_first=False
):
    # Interpolation is every num_frames, we need to create the inputs for the model
    # We need to create a list of inputs, each input is a tuple of ([first, last], audio)
    # The first and last are the first and last frames of the interpolation

    # Break video into chunks of num_frames with overlap 1
    # assert video.shape[0] == audio.shape[0], "Video and audio must have the same number of frames"
    audio_interpolation_chunks = []
    audio_image_preds = []
    gt_chunks = []
    emotions_chunks = []
    # Adjustment for overlap to ensure segments are created properly
    step = num_frames - overlap

    # raw_audio = rearrange(raw_audio, "... (f s) -> f s", s=640)

    # Ensure there's at least one step forward on each iteration
    if step < 1:
        step = 1
    # audio_image_preds.append(audio[0])
    audio_image_preds_idx = []
    audio_interp_preds_idx = []
    for i in range(0, audio.shape[0] - num_frames + 1, step):
        try:
            last = video[i + num_frames - 1]
        except IndexError:
            break  # Last chunk is smaller than num_frames
        segment_end = i + num_frames
        gt_chunks.append(video[i:segment_end])
        # audio_indexes = get_audio_indexes(segment_end - 1, 2, len(audio))
        # print(i, audio_indexes, 2 * additional_audio_frames + 1)
        if i not in audio_image_preds_idx:
            audio_image_preds.append(audio[i])
            audio_image_preds_idx.append(i)
            if emotions is not None:
                emotions_chunks.append((emotions[0][i], emotions[1][i]))
            # emotions_chunks.append((torch.ones_like(emotions[0][i]), torch.ones_like(emotions[1][i])))
        if segment_end - 1 not in audio_image_preds_idx:
            audio_image_preds_idx.append(segment_end - 1)
            audio_image_preds.append(audio[segment_end - 1])
            if emotions is not None:
                emotions_chunks.append((emotions[0][segment_end - 1], emotions[1][segment_end - 1]))
            # emotions_chunks.append(
            #     (torch.ones_like(emotions[0][segment_end - 1]), torch.ones_like(emotions[1][segment_end - 1]))
            # )
        audio_interpolation_chunks.append(audio_interpolation[i:segment_end])
        audio_interp_preds_idx.append([i, segment_end - 1])
        # raw_audio_chunks.append(raw_audio[i :segment_end])
        print(i, segment_end - 1)

    # If the flag is on, add element 0 every 14 audio elements
    if add_zero_flag:
        first_element = audio[0]
        if emotions is not None:
            first_element_emotions = (emotions[0][0], emotions[1][0])
        len_audio_image_preds = len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        for i in range(0, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, first_element)
            audio_image_preds_idx.insert(i, None)
            if emotions is not None:
                emotions_chunks.insert(i, first_element_emotions)

    # If double_first flag is activated, double the frame added every num_frames
    if double_first:
        print(len(audio_image_preds))
        len_audio_image_preds = len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        for i in range(0, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, audio_image_preds[i])
            audio_image_preds_idx.insert(i, None)
            if emotions is not None:
                emotions_chunks.insert(i, emotions_chunks[i])

    print(audio_image_preds_idx)
    to_remove = [idx is None for idx in audio_image_preds_idx]
    audio_image_preds_idx_clone = [idx for idx in audio_image_preds_idx]
    print(to_remove)
    if double_first or add_zero_flag:
        # Remove the added elements from the list
        audio_image_preds_idx = [sample for i, sample in zip(to_remove, audio_image_preds_idx) if not i]

    print(len(audio_image_preds_idx), audio_image_preds_idx)

    interpolation_cond_list = []
    for i in range(0, len(audio_image_preds_idx) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append([audio_image_preds_idx[i], audio_image_preds_idx[i + 1]])
    print(len(interpolation_cond_list), interpolation_cond_list)

    # Since we generate num_frames at a time, we need to ensure that the last chunk is of size num_frames
    # Calculate the number of frames needed to make audio_image_preds a multiple of num_frames
    frames_needed = (num_frames - (len(audio_image_preds) % num_frames)) % num_frames

    # Extend from the start of audio_image_preds
    audio_image_preds = audio_image_preds + [audio_image_preds[-1]] * frames_needed
    if emotions is not None:
        emotions_chunks = emotions_chunks + [emotions_chunks[-1]] * frames_needed
    to_remove = to_remove + [True] * frames_needed
    audio_image_preds_idx_clone = audio_image_preds_idx_clone + [audio_image_preds_idx_clone[-1]] * frames_needed

    print(f"Added {frames_needed} frames from the start to make audio_image_preds a multiple of {num_frames}")

    # random_cond_idx = np.random.randint(0, len(video_emb))
    random_cond_idx = 0

    assert len(to_remove) == len(audio_image_preds), "to_remove and audio_image_preds must have the same length"

    return (
        gt_chunks,
        audio_interpolation_chunks,
        audio_image_preds,
        video_emb[random_cond_idx],
        video[random_cond_idx],
        emotions_chunks,
        random_cond_idx,
        frames_needed,
        to_remove,
        audio_interp_preds_idx,
        audio_image_preds_idx_clone,
    )


def get_audio_embeddings(
    audio_path: str,
    audio_rate: int = 16000,
    fps: int = 25,
    audio_emb_type="wav2vec2",
    audio_folder=None,
    audio_emb_folder=None,
    extra_audio=False,
    max_frames=None,
):
    # Process audio
    audio = None
    raw_audio = None
    audio_interpolation = None
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
    elif audio_path is not None and audio_path.endswith(".safetensors"):
        audio = load_safetensors(audio_path)["audio"]
        if audio_emb_type != "wav2vec2":
            audio_interpolation = load_safetensors(audio_path.replace(f"_{audio_emb_type}_emb", "_wav2vec2_emb"))["audio"]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        print(audio.shape)
        if max_frames is not None:
            audio = audio[:max_frames]
            if audio_interpolation is not None:
                audio_interpolation = audio_interpolation[:max_frames]
        if extra_audio is not None:
            # extra_audio_emb = torch.load(audio_path.replace(f"_{audio_emb_type}_emb", "_beats_emb"))
            extra_audio_emb = load_safetensors(audio_path.replace(f"_{audio_emb_type}_emb", "_beats_emb"))["audio"]
            if max_frames is not None:
                extra_audio_emb = extra_audio_emb[:max_frames]
            print(
                f"Loaded extra audio embeddings from {audio_path.replace(f'_{audio_emb_type}_emb', '_beats_emb')} {extra_audio_emb.shape}."
            )
            min_size = min(audio.shape[0], extra_audio_emb.shape[0])
            audio = torch.cat([audio[:min_size], extra_audio_emb[:min_size]], dim=-1)
            print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")
        
        print(audio.shape)

        if audio_interpolation is None:
            audio_interpolation = audio
        print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")
        raw_audio_path = audio_path.replace(".safetensors", ".wav").replace(f"_{audio_emb_type}_emb", "")
        if audio_folder is not None:
            raw_audio_path = raw_audio_path.replace(audio_emb_folder, audio_folder)

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, audio_interpolation, raw_audio


def sample_keyframes(
    model_keyframes,
    audio_list,
    condition,
    num_frames,
    motion_bucket_id,
    fps_id,
    cond_aug,
    device,
    embbedings,
    valence_list,
    arousal_list,
    force_uc_zero_embeddings,
    n_batch_keyframes,
    added_frames,
    strength,
    scale,
    num_steps,
):
    if scale is not None:
        model_keyframes.sampler.guider.set_scale(scale)
    if num_steps is not None:
        model_keyframes.sampler.set_num_steps(num_steps)
    samples_list = []
    samples_z_list = []
    samples_x_list = []

    for i in range(audio_list.shape[0]):
        H, W = condition.shape[-2:]
        assert condition.shape[1] == 3
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

        audio_cond = audio_list[i].unsqueeze(0)

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
        if valence_list is not None:
            value_dict["valence"] = valence_list[i].unsqueeze(0)
            value_dict["arousal"] = arousal_list[i].unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model_keyframes.conditioner),
                    value_dict,
                    [1, 1],
                    T=num_frames,
                    device=device,
                )

                c, uc = model_keyframes.conditioner.get_unconditional_conditioning(
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

                for k in c:
                    if isinstance(c[k], torch.Tensor):
                        print(k, c[k].shape)
                print("video", video.shape)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch_keyframes, num_frames).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model_keyframes.denoiser(model_keyframes.model, input, sigma, c, **additional_model_inputs)

                samples_z = model_keyframes.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
                samples_z_list.append(samples_z)
                # interpolation_cond_list_emb.append(
                #     torch.stack([start_cond_emb.to(device), samples_z.squeeze(0).clone()], dim=1)
                # )
                # start_cond_emb = samples_z.squeeze(0).clone()

                samples_x = model_keyframes.decode_first_stage(samples_z)
                samples_x_list.append(samples_x)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                samples_list.append(samples)

                # interpolation_cond_list.append(torch.stack([start_cond.to(device), samples.squeeze(0).clone()], dim=1))
                # start_cond = samples.squeeze(0).clone()

                # image = samples[-1] * 2.0 - 1.0
                video = None

    samples = torch.concat(samples_list)[:-added_frames] if added_frames > 0 else torch.concat(samples_list)
    samples_z = torch.concat(samples_z_list)[:-added_frames] if added_frames > 0 else torch.concat(samples_z_list)
    samples_x = torch.concat(samples_x_list)[:-added_frames] if added_frames > 0 else torch.concat(samples_x_list)

    return samples_z, samples_x


def sample_interpolation(
    model,
    samples_z,
    samples_x,
    audio_interpolation_list,
    condition,
    num_frames,
    device,
    overlap,
    fps_id,
    motion_bucket_id,
    cond_aug,
    force_uc_zero_embeddings,
    n_batch,
    chunk_size,
    strength,
    scale,
    num_steps,
    cut_audio: bool = False,
    to_remove: list[int] = [],
):
    if scale is not None:
        model.sampler.guider.set_scale(scale)
    if num_steps is not None:
        model.sampler.set_num_steps(num_steps)
    # Creating condition for interpolation model. We need to create a list of inputs, each input is  [first, last]
    # The first and last are the first and last frames of the interpolation
    interpolation_cond_list = []
    interpolation_cond_list_emb = []

    # Remove zero embeddings if the flag is activated
    # if double_first:
    #     samples_x = [sample for i, sample in enumerate(samples_x) if (i) % (num_frames) != 0]
    #     samples_z = [sample for i, sample in enumerate(samples_z) if (i) % (num_frames) != 0]
    # elif add_zero_flag:
    #     samples_x = [sample for i, sample in enumerate(samples_x) if ((i) % (num_frames) != 0) or (i == 0)]
    #     samples_z = [sample for i, sample in enumerate(samples_z) if ((i) % (num_frames) != 0) or (i == 0)]
    samples_x = [sample for i, sample in zip(to_remove, samples_x) if not i]
    samples_z = [sample for i, sample in zip(to_remove, samples_z) if not i]

    for i in range(0, len(samples_z) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append(torch.stack([samples_x[i], samples_x[i + 1]], dim=1))
        interpolation_cond_list_emb.append(torch.stack([samples_z[i], samples_z[i + 1]], dim=1))

    condition = torch.stack(interpolation_cond_list).to(device)
    audio_cond = torch.stack(audio_interpolation_list).to(device)
    # condition = condition.unsqueeze(0).to(device)
    embbedings = torch.stack(interpolation_cond_list_emb).to(device)

    # condition = repeat(condition, "b c h w -> (b d) c h w", d=audio_cond.shape[0])
    # condition_emb = repeat(condition_emb, "b c h w -> (b d) c h w", d=audio_cond.shape[0])

    H, W = condition.shape[-2:]
    # assert condition.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames * audio_cond.shape[0], C, H // F, W // F)
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

    value_dict["cond_frames"] = embbedings
    value_dict["cond_aug"] = cond_aug
    if cut_audio:
        value_dict["audio_emb"] = audio_cond[:, :, :, :768]
    else:
        value_dict["audio_emb"] = audio_cond
    print(value_dict["audio_emb"].shape)
    # value_dict["gt"] = rearrange(embbedings, "b t c h w -> b c t h w").to(device)
    # masked_gt = value_dict["gt"] * (1 - value_dict["masks"])

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch_overlap(
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

            # n_batch *= embbedings.shape[0]

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch, num_frames).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            print(condition.shape, embbedings.shape, audio_cond.shape, shape, additional_model_inputs)

            if chunk_size is not None:
                chunk_size = chunk_size * num_frames

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                    num_overlap_frames=overlap,
                    num_frames=num_frames,
                    n_skips=n_batch,
                    chunk_size=chunk_size,
                    **additional_model_inputs,
                )

            samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)

            samples_x = model.decode_first_stage(samples_z)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            video = None

    samples = rearrange(samples, "(b t) c h w -> b t c h w", t=num_frames)
    samples = merge_overlapping_segments(samples, overlap)

    vid = (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)

    # gt_vid = (rearrange(gt, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
    # for frame in vid:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     writer.write(frame)
    # writer.release()
    # if raw_audio is not None:
    #     raw_audio = rearrange(raw_audio[: len(vid)], "f s -> () (f s)")

    return vid


def sample(
    model,
    model_keyframes,
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
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
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
    model_keyframes_config: Optional[str] = None,
    max_seconds: Optional[int] = None,
    min_seconds: Optional[int] = None,
    lora_path_interp: Optional[str] = None,
    lora_path_keyframes: Optional[str] = None,
    force_uc_zero_embeddings=[
        "cond_frames",
        "cond_frames_without_noise",
    ],
    get_landmarks: bool = False,
    chunk_size: int = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    is_dub: bool = False,
    keyframes_ckpt: Optional[str] = None,
    interpolation_ckpt: Optional[str] = None,
    add_zero_flag: bool = False,
    recurse: bool = False,
    double_first: bool = False,
    n_batch: int = 1,
    n_batch_keyframes: int = 1,
    compute_until: float = "end",
    extra_audio: bool = False,
    audio_emb_type:str="wav2vec2"
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    assert not (double_first and add_zero_flag), "Cannot have both double_first and add_zero_flag"

    if version == "svd":
        num_frames = default(num_frames, 14)
        # num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/full_pipeline/svd/")
        # model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        # num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/full_pipeline/svd_xt/")
        # model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        # num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/full_pipeline/svd_image_decoder/")
        # model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        # num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/full_pipeline/svd_xt_image_decoder/")
        # model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    os.makedirs(output_folder, exist_ok=True)

    # base_count = len(glob(os.path.join(output_folder, "*.mp4")))

    out_video_path = os.path.join(output_folder, os.path.basename(video_path))

    if os.path.exists(out_video_path):
        print(f"Video already exists at {out_video_path}. Skipping.")
        return

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
        h, w = video.shape[2:]
        video = torch.nn.functional.interpolate(video, (512, 512), mode="bilinear")

        video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
        if video_folder is not None and latent_folder is not None:
            video_embedding_path = video_embedding_path.replace(video_folder, latent_folder)
        video_emb = None
        if use_latent:
            video_emb = load_safetensors(video_embedding_path)["latents"].cpu()

        if compute_until == "end":
            compute_until = int((video.shape[0] * 10) / 25)

        if compute_until is not None:
            max_frames = compute_until * fps_id
        print("max_frames", max_frames)
        audio, audio_interpolation, raw_audio = get_audio_embeddings(
            audio_path,
            16000,
            fps_id + 1,
            audio_folder=audio_folder,
            audio_emb_folder=audio_emb_folder,
            extra_audio=extra_audio,
            max_frames=max_frames,
            audio_emb_type=audio_emb_type
        )
        if compute_until is not None:
            if video.shape[0] > max_frames:
                video = video[:max_frames]
                audio = audio[:max_frames]
                video_emb = video_emb[:max_frames] if video_emb is not None else None
                raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
        if min_seconds is not None:
            min_frames = min_seconds * fps_id
            video = video[min_frames:]
            audio = audio[min_frames:]
            video_emb = video_emb[min_frames:] if video_emb is not None else None
            raw_audio = raw_audio[min_frames:] if raw_audio is not None else None
        audio = audio

        print("Video has ", video.shape[0], "frames", "and", video.shape[0] / 25, "seconds")

        emotions = None
        if emotion_folder is not None:
            emotions_path = video_path.replace(video_folder, emotion_folder).replace(".mp4", ".pt")
            emotions = torch.load(emotions_path)
            emotions = emotions["valence"], emotions["arousal"]

        landmarks = None
        if get_landmarks:
            if landmark_folder is not None:
                video_path = video_path.replace(video_folder, landmark_folder)
            landmarks = np.load(
                video_path.replace(".mp4", ".npy").replace("_output_output", "_output_keypoints"), allow_pickle=True
            )
            landmarks = scale_landmarks(landmarks[:, :, :2], (h, w), (512, 512))

        h, w = video.shape[2:]
        # if degradation > 1:
        #     video = torch.nn.functional.interpolate(video, (h // degradation, w // degradation), mode="bilinear")
        #     video = torch.nn.functional.interpolate(video, (h, w), mode="bilinear")
        #     # Save video
        #     out_path = os.path.join(output_folder, "degraded.mp4")
        #     write_video(out_path, (((video.permute(0, 2, 3, 1) + 1) / 2) * 255.0).cpu(), fps_id)

        if input_img_path is None or input_img_path == "":
            model_input = video
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

        (
            gt_chunks,
            audio_interpolation_list,
            audio_list,
            emb,
            cond,
            emotions_chunks,
            _,
            added_frames,
            to_remove,
            test_interpolation_list,
            test_keyframes_list,
        ) = create_pipeline_inputs(
            model_input,
            audio,
            audio_interpolation,
            num_frames,
            video_emb,
            emotions,
            overlap=overlap,
            add_zero_flag=add_zero_flag,
            double_first=double_first,
        )

        print(n_batch, n_batch_keyframes)

        if lora_path_keyframes is not None:
            model_keyframes.init_from_ckpt(lora_path_keyframes, remove_keys_from_weights=None)

        model_keyframes.en_and_decode_n_samples_a_time = decoding_t
        model.en_and_decode_n_samples_a_time = decoding_t

        additional_audio_frames = model_keyframes.model.diffusion_model.additional_audio_frames
        print(f"Additional audio frames: {additional_audio_frames}")

        audio_list = torch.stack(audio_list).to(device)
        audio_list = rearrange(audio_list, "(b t) c d  -> b t c d", t=num_frames)

        # Convert to_remove into chunks of num_frames
        to_remove_chunks = [to_remove[i : i + num_frames] for i in range(0, len(to_remove), num_frames)]
        test_keyframes_list = [
            test_keyframes_list[i : i + num_frames] for i in range(0, len(test_keyframes_list), num_frames)
        ]

        valence_list = None
        arousal_list = None
        if emotions is not None:
            valence_list = torch.stack([x[0] for x in emotions_chunks]).to(device)
            arousal_list = torch.stack([x[1] for x in emotions_chunks]).to(device)
            valence_list = rearrange(valence_list, "(b t) -> b t", t=num_frames)
            arousal_list = rearrange(arousal_list, "(b t) -> b t", t=num_frames)

        condition = cond

        audio_cond = audio_list
        condition = condition.unsqueeze(0).to(device)
        embbedings = emb.unsqueeze(0).to(device) if emb is not None else None

        # One batch of keframes is approximately 7 seconds
        chunk_size = max_seconds // 7
        complete_video = []
        complete_audio = []
        start_idx = 0
        last_frame_z = None
        last_frame_x = None
        last_keyframe_idx = None
        last_to_remove = None
        for chunk_start in range(0, len(audio_cond), chunk_size):
            # is_last_chunk = chunk_start + chunk_size >= len(audio_cond)

            chunk_end = min(chunk_start + chunk_size, len(audio_cond))

            chunk_audio_cond = audio_cond[chunk_start:chunk_end].cuda()
            chunk_valence_list = valence_list[chunk_start:chunk_end].cuda() if valence_list is not None else None
            chunk_arousal_list = arousal_list[chunk_start:chunk_end].cuda() if arousal_list is not None else None

            # # Free up some memory from the keyframes
            # del model_keyframes
            # torch.cuda.empty_cache()
            test_keyframes_list_unwrapped = [
                elem for sublist in test_keyframes_list[chunk_start:chunk_end] for elem in sublist
            ]
            to_remove_chunks_unwrapped = [
                elem for sublist in to_remove_chunks[chunk_start:chunk_end] for elem in sublist
            ]

            if last_keyframe_idx is not None:
                test_keyframes_list_unwrapped = [last_keyframe_idx] + test_keyframes_list_unwrapped
                to_remove_chunks_unwrapped = [last_to_remove] + to_remove_chunks_unwrapped

            last_keyframe_idx = test_keyframes_list_unwrapped[-1]
            last_to_remove = to_remove_chunks_unwrapped[-1]
            # Find the first non-None keyframe in the chunk
            first_keyframe = next((kf for kf in test_keyframes_list_unwrapped if kf is not None), None)

            # Find the last non-None keyframe in the chunk
            last_keyframe = next((kf for kf in reversed(test_keyframes_list_unwrapped) if kf is not None), None)

            start_idx = next(
                (idx for idx, comb in enumerate(test_interpolation_list) if comb[0] == first_keyframe), None
            )
            end_idx = next(
                (idx for idx, comb in enumerate(reversed(test_interpolation_list)) if comb[1] == last_keyframe), None
            )

            if start_idx is not None and end_idx is not None:
                end_idx = len(test_interpolation_list) - 1 - end_idx  # Adjust for reversed enumeration
            end_idx += 1

            if end_idx < start_idx:
                end_idx = len(audio_interpolation_list)

            audio_interpolation_list_chunk = audio_interpolation_list[start_idx:end_idx]

            print(start_idx, end_idx)
            print("Testing keyframes: ", test_keyframes_list_unwrapped)
            print("Testing interpolation: ", test_interpolation_list[start_idx:end_idx])
            print("To remove: ", to_remove_chunks_unwrapped)

            samples_z, samples_x = sample_keyframes(
                model_keyframes,
                chunk_audio_cond,
                condition.cuda(),
                num_frames,
                motion_bucket_id,
                fps_id,
                cond_aug,
                device,
                embbedings.cuda(),
                chunk_valence_list,
                chunk_arousal_list,
                force_uc_zero_embeddings,
                n_batch_keyframes,
                0,
                strength,
                None,
                None,
            )

            # print("sample_z shape", samples_z.shape)
            # print("samples_x shape", samples_x.shape)
            # print("is_last_chunk", is_last_chunk)
            # print("added_frames", added_frames)
            # print("audio_interpolation_list_chunk", len(audio_interpolation_list_chunk))
            # print(to_remove_chunks[chunk_start:chunk_end])

            # audio_vid = torch.cat(raw_audio_chunks[start_idx:end_idx])

            if last_frame_x is not None:
                samples_x = torch.cat([last_frame_x.unsqueeze(0), samples_x], axis=0)
                samples_z = torch.cat([last_frame_z.unsqueeze(0), samples_z], axis=0)

            last_frame_x = samples_x[-1]
            last_frame_z = samples_z[-1]

            vid = sample_interpolation(
                model,
                samples_z,
                samples_x,
                audio_interpolation_list_chunk,
                condition.cuda(),
                num_frames,
                device,
                overlap,
                fps_id,
                motion_bucket_id,
                cond_aug,
                force_uc_zero_embeddings,
                n_batch,
                chunk_size,
                strength,
                None,
                None,
                cut_audio=extra_audio != "both",
                to_remove=to_remove_chunks_unwrapped,
            )

            if chunk_start == 0:
                complete_video = vid
            else:
                complete_video = np.concatenate([complete_video[:-1], vid], axis=0)

        if raw_audio is not None:
            complete_audio = rearrange(raw_audio[: complete_video.shape[0]], "f s -> () (f s)")

        save_audio_video(
            complete_video,
            audio=complete_audio,
            frame_rate=fps_id + 1,
            sample_rate=16000,
            save_path=out_video_path,
            keep_intermediate=False,
        )

    print(f"Saved video to {out_video_path}")


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


def get_batch_overlap(keys, value_dict, N, T, device):
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
            batch[key] = repeat(value_dict["cond_frames"], "b ... -> (b t) ...", t=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "b ... -> (b t) ...", t=N[0])
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
    ckpt: Optional[str] = None,
    low_sigma: float = 0.0,
    high_sigma: float = float("inf"),
):
    config = OmegaConf.load(config)
    # if device == "cuda":
    #     config.model.params.conditioner_config.params.emb_models[
    #         0
    #     ].params.open_clip_embedding_config.params.init_device = device

    config["model"]["params"]["input_key"] = input_key

    if ckpt is not None:
        config.model.params.ckpt_path = ckpt

    if num_steps is not None:
        config.model.params.sampler_config.params.num_steps = num_steps
    if "num_frames" in config.model.params.sampler_config.params.guider_config.params:
        config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    # Add low_sigma and high_sigma to the sampler config
    # config.model.params.sampler_config.params.low_sigma = low_sigma
    # config.model.params.sampler_config.params.high_sigma = high_sigma

    if "IdentityGuider" in config.model.params.sampler_config.params.guider_config.target:
        n_batch = 1
    elif "MultipleCondVanilla" in config.model.params.sampler_config.params.guider_config.target:
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


def main(
    input_path: str = "",  # Can either be image file or folder with image files
    filelist: str = "",
    filelist_audio: str = "",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    landmark_folder: Optional[str] = None,
    emotion_folder: Optional[str] = None,
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
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
    model_keyframes_config: Optional[str] = None,
    max_seconds: Optional[int] = None,
    min_seconds: Optional[int] = None,
    lora_path_interp: Optional[str] = None,
    lora_path_keyframes: Optional[str] = None,
    force_uc_zero_embeddings=[
        "cond_frames",
        "cond_frames_without_noise",
    ],
    get_landmarks: bool = False,
    chunk_size: int = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    is_dub: bool = False,
    keyframes_ckpt: Optional[str] = None,
    interpolation_ckpt: Optional[str] = None,
    add_zero_flag: bool = False,
    recurse: bool = False,
    double_first: bool = False,
    extra_audio: str = None,
    compute_until: str = "end",
    starting_index: int = 0,
    audio_emb_type: str="wav2vec2"
):
    num_frames = default(num_frames, 14)
    model, filter, n_batch = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        "latents",
        interpolation_ckpt,
    )

    if lora_path_interp is not None:
        model.init_from_ckpt(lora_path_interp, remove_keys_from_weights=None)

    model_keyframes, filter, n_batch_keyframes = load_model(
        model_keyframes_config,
        device,
        num_frames,
        num_steps,
        "latents",
        keyframes_ckpt,
    )

    # Open the filelist and read the video paths
    with open(filelist, "r") as f:
        video_paths = f.readlines()

    # Remove the newline character from each path
    video_paths = [path.strip() for path in video_paths]

    if filelist_audio:
        with open(filelist_audio, "r") as f:
            audio_paths = f.readlines()
        audio_paths = [
            path.strip().replace("video_crop", "audio_emb").replace(".mp4", f"_{audio_emb_type}_emb.safetensors") for path in audio_paths
        ]
    else:
        audio_paths = [
            video_path.replace("video_crop", "audio_emb").replace(".mp4", f"_{audio_emb_type}_emb.safetensors")
            for video_path in video_paths
        ]

    if starting_index:
        video_paths = video_paths[starting_index:]
        audio_paths = audio_paths[starting_index:]

    for video_path, audio_path in zip(video_paths, audio_paths):
        sample(
            model,
            model_keyframes,
            video_path=video_path,
            audio_path=audio_path,
            num_frames=num_frames,
            num_steps=num_steps,
            resize_size=resize_size,
            video_folder=video_folder,
            latent_folder=latent_folder,
            landmark_folder=landmark_folder,
            emotion_folder=emotion_folder,
            audio_folder=audio_folder,
            audio_emb_folder=audio_emb_folder,
            version=version,
            fps_id=fps_id,
            motion_bucket_id=motion_bucket_id,
            cond_aug=cond_aug,
            seed=seed,
            decoding_t=decoding_t,
            device=device,
            output_folder=output_folder,
            autoregressive=autoregressive,
            strength=strength,
            use_latent=use_latent,
            model_config=model_config,
            model_keyframes_config=model_keyframes_config,
            max_seconds=max_seconds,
            min_seconds=min_seconds,
            lora_path_interp=lora_path_interp,
            lora_path_keyframes=lora_path_keyframes,
            force_uc_zero_embeddings=force_uc_zero_embeddings,
            get_landmarks=get_landmarks,
            chunk_size=chunk_size,
            overlap=overlap,
            is_dub=is_dub,
            keyframes_ckpt=keyframes_ckpt,
            interpolation_ckpt=interpolation_ckpt,
            add_zero_flag=add_zero_flag,
            recurse=recurse,
            double_first=double_first,
            n_batch=n_batch,
            n_batch_keyframes=n_batch_keyframes,
            extra_audio=extra_audio,
            compute_until=compute_until,
            audio_emb_type=audio_emb_type
        )


if __name__ == "__main__":
    Fire(main)
