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
    video, audio, num_frames, video_emb, emotions=None, overlap=1, add_zero_flag=False, double_first=False
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

    # Ensure there's at least one step forward on each iteration
    if step < 1:
        step = 1
    # audio_image_preds.append(audio[0])
    audio_image_preds_idx = []
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
        audio_interpolation_chunks.append(audio[i:segment_end])
        print(i, segment_end - 1)

    # If the flag is on, add element 0 every 14 audio elements
    print(len(audio_image_preds), len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames)
    print(len(audio_image_preds_idx), len(audio_image_preds))
    print(audio_image_preds_idx)
    if add_zero_flag:
        first_element = audio[0]
        if emotions is not None:
            first_element_emotions = (emotions[0][0], emotions[1][0])
        len_audio_image_preds = len(audio_image_preds)
        for i in range(num_frames, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, first_element)
            audio_image_preds_idx.insert(i, audio_image_preds_idx[0])
            if emotions is not None:
                emotions_chunks.insert(i, first_element_emotions)

    # If double_first flag is activated, double the frame added every num_frames
    if double_first:
        print(len(audio_image_preds))
        len_audio_image_preds = len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        for i in range(0, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, audio_image_preds[i])
            audio_image_preds_idx.insert(i, audio_image_preds_idx[i])
            if emotions is not None:
                emotions_chunks.insert(i, emotions_chunks[i])

    print(audio_image_preds_idx)
    if double_first:
        # Remove the added elements from the list
        audio_image_preds_idx = [sample for i, sample in enumerate(audio_image_preds_idx) if (i) % (num_frames) != 0]
    elif add_zero_flag:
        audio_image_preds_idx = [
            sample for i, sample in enumerate(audio_image_preds_idx) if ((i) % (num_frames) != 0) or (i == 0)
        ]
    print(audio_image_preds_idx)

    interpolation_cond_list = []
    for i in range(0, len(audio_image_preds_idx) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append([audio_image_preds_idx[i], audio_image_preds_idx[i + 1]])

    # Since we generate num_frames at a time, we need to ensure that the last chunk is of size num_frames
    # Calculate the number of frames needed to make audio_image_preds a multiple of num_frames
    frames_needed = (num_frames - (len(audio_image_preds) % num_frames)) % num_frames

    # Extend from the start of audio_image_preds
    audio_image_preds = audio_image_preds + audio_image_preds[:frames_needed]
    if emotions is not None:
        emotions_chunks = emotions_chunks + emotions_chunks[:frames_needed]

    print(f"Added {frames_needed} frames from the start to make audio_image_preds a multiple of {num_frames}")

    # random_cond_idx = np.random.randint(0, len(video_emb))
    random_cond_idx = 0

    return (
        gt_chunks,
        audio_interpolation_chunks,
        audio_image_preds,
        video_emb[random_cond_idx],
        video[random_cond_idx],
        emotions_chunks,
        random_cond_idx,
        frames_needed,
    )


def get_audio_embeddings(
    audio_path: str,
    audio_rate: int = 16000,
    fps: int = 25,
    audio_emb_type="wav2vec2",
    audio_folder=None,
    audio_emb_folder=None,
    extra_audio: bool = False,
    max_frames: int = None,
):
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
        if max_frames is not None:
            audio = audio[:max_frames]
        if extra_audio:
            extra_audio_emb = torch.load(audio_path.replace(f"_{audio_emb_type}_emb", "_beats_emb"))
            if max_frames is not None:
                extra_audio_emb = extra_audio_emb[:max_frames]
            print(
                f"Loaded extra audio embeddings from {audio_path.replace(f'_{audio_emb_type}_emb', '_beats_emb')} {extra_audio_emb.shape}."
            )
            min_size = min(audio.shape[0], extra_audio_emb.shape[0])
            audio = torch.cat([audio[:min_size], extra_audio_emb[:min_size]], dim=-1)
            print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")

        print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")
        raw_audio_path = audio_path.replace(".pt", ".wav").replace(f"_{audio_emb_type}_emb", "")
        if audio_folder is not None:
            raw_audio_path = raw_audio_path.replace(audio_emb_folder, audio_folder)

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, raw_audio


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
    keyframe_path,
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

    samples = torch.concat(samples_list)[:-added_frames]
    samples_z = torch.concat(samples_z_list)[:-added_frames]
    samples_x = torch.concat(samples_x_list)[:-added_frames]

    keyframes_gen = (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)

    save_audio_video(
        keyframes_gen,
        audio=None,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=keyframe_path,
        keep_intermediate=False,
    )

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
    add_zero_flag,
    raw_audio,
    fps_id,
    motion_bucket_id,
    cond_aug,
    force_uc_zero_embeddings,
    n_batch,
    chunk_size,
    strength,
    gt_chunks,
    video_path,
    video_path_gt,
    scale,
    num_steps,
    double_first,
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
    if double_first:
        samples_x = [sample for i, sample in enumerate(samples_x) if (i) % (num_frames) != 0]
        samples_z = [sample for i, sample in enumerate(samples_z) if (i) % (num_frames) != 0]
    elif add_zero_flag:
        samples_x = [sample for i, sample in enumerate(samples_x) if ((i) % (num_frames) != 0) or (i == 0)]
        samples_z = [sample for i, sample in enumerate(samples_z) if ((i) % (num_frames) != 0) or (i == 0)]

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
    value_dict["audio_emb"] = audio_cond[:, :, :, :768]
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

    gt_chunks = torch.stack(gt_chunks)
    gt_chunks = merge_overlapping_segments(gt_chunks, overlap)
    gt_chunks = torch.clamp((gt_chunks + 1.0) / 2.0, min=0.0, max=1.0)
    gt_vid = (gt_chunks * 255).cpu().numpy().astype(np.uint8)

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
        raw_audio = rearrange(raw_audio[: len(vid)], "f s -> () (f s)")

    save_audio_video(
        vid,
        audio=raw_audio,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=video_path,
        keep_intermediate=False,
    )

    save_audio_video(
        gt_vid,
        audio=raw_audio,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=video_path_gt,
        keep_intermediate=False,
    )

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
    extra_audio: bool = False,
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

    if use_latent:
        input_key = "latents"
    else:
        input_key = "frames"

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
            video_emb = load_safetensors(video_embedding_path)["latents"]

        if max_seconds is not None:
            max_frames = max_seconds * fps_id
        audio, raw_audio = get_audio_embeddings(
            audio_path,
            16000,
            fps_id + 1,
            audio_folder=audio_folder,
            audio_emb_folder=audio_emb_folder,
            extra_audio=extra_audio,
            max_frames=max_frames,
        )
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
            emotions = emotions["valence"][:max_frames], emotions["arousal"][:max_frames]

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

        gt_chunks, audio_interpolation_list, audio_list, emb, cond, emotions_chunks, _, added_frames = (
            create_pipeline_inputs(
                model_input,
                audio,
                num_frames,
                video_emb,
                emotions,
                overlap=overlap,
                add_zero_flag=add_zero_flag,
                double_first=double_first,
            )
        )

        model, filter, n_batch = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
            input_key,
            interpolation_ckpt,
        )

        if lora_path_interp is not None:
            model.init_from_ckpt(lora_path_interp, remove_keys_from_weights=None)

        model_keyframes, filter, n_batch_keyframes = load_model(
            model_keyframes_config,
            device,
            num_frames,
            num_steps,
            input_key,
            keyframes_ckpt,
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

        valence_list = None
        arousal_list = None
        if emotions is not None:
            valence_list = torch.stack([x[0] for x in emotions_chunks]).to(device)
            arousal_list = torch.stack([x[1] for x in emotions_chunks]).to(device)
            valence_list = rearrange(valence_list, "(b t) -> b t", t=num_frames)
            arousal_list = rearrange(arousal_list, "(b t) -> b t", t=num_frames)

        condition = cond

        audio_cond = audio_list.cuda()
        condition = condition.unsqueeze(0).to(device)
        embbedings = emb.unsqueeze(0).to(device) if emb is not None else None

        os.makedirs(output_folder, exist_ok=True)

        if recurse:
            while True:
                scale = input("Enter the scale value (or 'stop' to end): ")
                if scale.lower() == "stop":
                    break
                try:
                    scale = float(scale)
                except ValueError:
                    print("Invalid input. Please enter a number or 'stop'.")
                    continue

                scale_interpolation = input(
                    "Enter the scale value for interpolation (or press Enter to use default): "
                )
                if scale_interpolation:
                    try:
                        scale_interpolation = float(scale_interpolation)
                    except ValueError:
                        print("Invalid input for scale_interpolation. Using default.")

                steps = input("Enter the number of steps (or press Enter to use default): ")
                if steps:
                    try:
                        steps = int(steps)
                    except ValueError:
                        print("Invalid input for steps. Using default.")
                        steps = None
                else:
                    steps = None

                steps_interpolation = input(
                    "Enter the number of steps for interpolation (or press Enter to use default): "
                )
                if steps_interpolation:
                    try:
                        steps_interpolation = int(steps_interpolation)
                    except ValueError:
                        print("Invalid input for steps_interpolation. Using default.")
                        steps_interpolation = None
                else:
                    steps_interpolation = None

                base_count = len(glob(os.path.join(output_folder, "*.mp4")))

                keyframe_path = os.path.join(
                    output_folder,
                    f"{base_count:06d}_keyframes_scale_{scale}_steps_{steps}.mp4",
                )
                video_path = os.path.join(
                    output_folder, f"{base_count:06d}_scale_{scale_interpolation}_steps_{steps_interpolation}.mp4"
                )
                video_path_gt = os.path.join(output_folder, f"{base_count:06d}_gt.mp4")

                samples_z, samples_x = sample_keyframes(
                    model_keyframes,
                    audio_cond,
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
                    keyframe_path,
                    scale=scale,
                    num_steps=steps,
                )

                # Free up some memory from the keyframes
                # del model_keyframes
                # torch.cuda.empty_cache()

                sample_interpolation(
                    model,
                    samples_z,
                    samples_x,
                    audio_interpolation_list,
                    condition,
                    num_frames,
                    device,
                    overlap,
                    add_zero_flag,
                    raw_audio,
                    fps_id,
                    motion_bucket_id,
                    cond_aug,
                    force_uc_zero_embeddings,
                    n_batch,
                    chunk_size,
                    strength,
                    gt_chunks,
                    video_path,
                    video_path_gt,
                    scale_interpolation,
                    steps_interpolation,
                    double_first,
                )
        else:
            base_count = len(glob(os.path.join(output_folder, "*.mp4")))

            keyframe_path = os.path.join(output_folder, f"{base_count:06d}_keyframes.mp4")
            video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
            video_path_gt = os.path.join(output_folder, f"{base_count:06d}_gt.mp4")
            samples_z, samples_x = sample_keyframes(
                model_keyframes,
                audio_cond,
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
                keyframe_path,
                None,
                None,
            )
            # Free up some memory from the keyframes
            del model_keyframes
            torch.cuda.empty_cache()

            sample_interpolation(
                model,
                samples_z,
                samples_x,
                audio_interpolation_list,
                condition,
                num_frames,
                device,
                overlap,
                add_zero_flag,
                raw_audio,
                fps_id,
                motion_bucket_id,
                cond_aug,
                force_uc_zero_embeddings,
                n_batch,
                chunk_size,
                strength,
                gt_chunks,
                video_path,
                video_path_gt,
                None,
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


if __name__ == "__main__":
    Fire(sample)
