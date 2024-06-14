import math
import os
from glob import glob
from pathlib import Path
from typing import Optional
from tqdm import tqdm  # Correct import

import face_alignment
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
import torchaudio
from safetensors.torch import load_file as load_safetensors
import torch.nn.functional as F
from torchvision.io import read_video

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio, save_audio_video
from sgm.data.data_utils import (
	create_masks_from_landmarks_full_size,
	create_face_mask_from_landmarks,
	create_masks_from_landmarks_box,
)
from scripts.util.audio_wrapper import AudioWrapper
from scripts.util.vae_wrapper import VaeWrapper
from scripts.util.utilities import *


"""
python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
--video_path=/data/home/stellab/projects/dubbing_gans/results_demo/insta_5/3345252620443890114_0_chain0_translated.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/WDA_BarackObama_000_wav2vec2_emb.pt \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=10 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10


	/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4

	--audio_path=/fsx/rs2517/data/HDTF/audio/RD_Radio18_000_wav2vec2_emb.pt \
	--audio_path=/data/home/stellab/projects/dubbing_gans/results_demo/evaluation_data/kimmel.wav \
	--audio_path=/fsx/rs2517/data/HDTF/audio/WDA_BarackObama_000_wav2vec2_emb.pt
	   --video_path=/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4 \
	 --video_path=./A043_C047_0721BS_001_output_output.mp4 \ WDA_HillaryClinton_000

python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/rs2517/data/HDTF/cropped_videos_original/WDA_BarackObama_000.mp4 \
	--audio_path=/data/home/stellab/projects/dubbing_gans/results_demo/evaluation_data/kimmel.wav
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=10 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10

	python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/rs2517/data/HDTF/cropped_videos_original/WDA_BarackObama_000.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/RD_Radio18_000_wav2vec2_emb.pt \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=5 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=jawline --overlap=5 --chunk_size=10


	nik_eng_heygen WDA_BarackObama_000 /fsx/behavioural_computing_data/face_generation_data/AA_processed/part_27/video_aligned_512/A043_C047_0721BS_001_output_output.mp4

python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/rs2517/data/HDTF/cropped_videos_original/RD_Radio18_000.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/RD_Radio18_000.wav \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=10 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=jawline --overlap=5 --chunk_size=10
"""


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


def create_interpolation_inputs(video, audio, landmarks, num_frames, video_emb=None, overlap=1, what_mask="full"):
	assert video.shape[0] == audio.shape[0], "Video and audio must have the same number of frames"
	masks_chunks = []
	audio_chunks = []
	video_emb_chunks = []
	gt_chunks = []
	masks_chunks_big = []
	# Adjustment for overlap to ensure segments are created properly
	step = num_frames - overlap

	# Ensure there's at least one step forward on each iteration
	if step < 1:
		step = 1
	
	for i in range(0, video.shape[0] - num_frames + 1, step):
		segment_end = i + num_frames
		if what_mask == "full":
			masks = create_masks_from_landmarks_full_size(
				landmarks[i:segment_end, :], video.shape[-1], video.shape[-2], offset=-0.01
			)
		elif what_mask == "box":
			masks = create_masks_from_landmarks_box(landmarks[i:segment_end, :], (video.shape[-1], video.shape[-2]))
		else:
			masks = create_face_mask_from_landmarks(
				landmarks[i:segment_end, :],
				video.shape[-1],
				video.shape[-2],
				mask_expand=0.05,
			)
		masks_chunks_big.append(masks)
		gt_chunks.append(video[i:segment_end])
		masks = F.interpolate(masks.unsqueeze(1).float(), size=(64, 64), mode="nearest")
		masks_chunks.append(masks)
		if video_emb is not None:
			video_emb_chunks.append(video_emb[i:segment_end])
		if audio is not None:
			audio_chunks.append(audio[i:segment_end])

	return gt_chunks, masks_chunks, audio_chunks, video_emb_chunks, masks_chunks_big


def get_audio_embeddings(audio_path: str, audio_rate: int = 16000, fps: int = 25, audio_model = None, save_emb = None):
	# Process audio
	audio = None
	raw_audio = None
	if audio_path is not None and (audio_path.endswith(".wav") or audio_path.endswith(".mp3")):
		audio, sr = torchaudio.load(audio_path, channels_first=True)
		if audio.shape[0] > 1:
			audio = audio.mean(0, keepdim=True)
		if sr != audio_rate:
			audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=audio_rate)[0]
			audio = audio.unsqueeze(0)

		samples_per_frame = math.ceil(audio_rate / fps)
		# audio_zeros = torch.zeros(audio.shape)
		# audio = audio_zeros
		raw_audio = audio.clone()
		raw_audio = raw_audio.squeeze(0)
		n_frames = raw_audio.shape[-1] / samples_per_frame
		if not n_frames.is_integer():
			# print("Audio shape before trim_pad_audio: ", raw_audio.shape)
			raw_audio = trim_pad_audio(raw_audio, audio_rate, max_len_raw=math.ceil(n_frames) * samples_per_frame)
			# print("Audio shape after trim_pad_audio: ", raw_audio.shape)
		raw_audio = rearrange(raw_audio, "(f s) -> f s", s=samples_per_frame)

	   
		if audio_model is not None:
			audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
			audio_embeddings = audio_model.encode_audio(audio)
			audio_file_name = get_name_from_file(audio_path)
			if save_emb is not None:
				torch.save(audio_embeddings.squeeze(0).cpu(), os.path.join(save_emb, '{}.pt'.format(audio_file_name)))
		else:
			print('Load audio model')
			exit()

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
		audio_embeddings = torch.load(audio_path)
		raw_audio_path = audio_path.replace(".pt", ".wav").replace("_whisper_emb", "").replace("_wav2vec2_emb", "")

		if os.path.exists(raw_audio_path):
			raw_audio = get_raw_audio(raw_audio_path, audio_rate)
		else:
			print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")
	return audio_embeddings, raw_audio

def encode_video(video, vae_model, device):
	n_samples = 14
	video_ = video.clone()
	video_ = torch.nn.functional.interpolate(video_.float(), size=512, mode="bilinear", align_corners=False).to(device)
	n_rounds = math.ceil(video_.shape[0] / n_samples)
	video_ = rearrange(video_, "t c h w -> c t h w")
	all_out = []
	with torch.no_grad():
		for n in tqdm(range(n_rounds)):         
			chunk_video = video_[:, n * n_samples : (n + 1) * n_samples, :, :]
			out = vae_model.encode_video(chunk_video.unsqueeze(0)).squeeze(0) # torch.Size([1, 4, 14, 64, 64])
			out = rearrange(out, "c t h w -> t c h w")
			all_out.append(out)      
	z = torch.cat(all_out, dim=0)
	return z

def sample(
	video_path: Optional[str] = None,
	audio_path: Optional[str] = None,  # Path to precomputed embeddings
	num_frames: Optional[int] = None,  # No need to touch
	num_steps: Optional[int] = None,  # Num steps diffusion process
	resize_size: Optional[int] = None,  # Resize image to this size
	version: str = "svd",
	fps_id: int = 24,  # Not use here
	motion_bucket_id: int = 127,  # Not used here
	cond_aug: float = 0.0,  # Not used here
	seed: int = 23,
	decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
	device: str = "cuda",
	output_folder: Optional[str] = None,
	strength: float = 1.0,  # Not used here
	use_latent: bool = False,  # If need to input to be latent
	# degradation: int = 1,
	overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
	what_mask: str = "full",  # Type of mask to use
	model_config: Optional[str] = None,
	max_seconds: Optional[int] = None,  # Max seconds of video to generate (HDTF if pretty long so better to limit)
	lora_path: Optional[str] = None,  # Not needed
	force_uc_zero_embeddings=[
		"cond_frames",
		"cond_frames_without_noise",
	],  # Useful for the classifier free guidance. What should be zeroed out in the unconditional embeddings
	chunk_size: int = None,  # Useful if the model gets OOM
):
	"""
	Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
	image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
	"""

	if version == "svd":
		num_frames = default(num_frames, 14)
		num_steps = default(num_steps, 25)
		output_folder = default(output_folder, "outputs/simple_dub_sample/svd/")
		# model_config = "scripts/sampling/configs/svd.yaml"
	elif version == "svd_xt":
		num_frames = default(num_frames, 25)
		num_steps = default(num_steps, 30)
		output_folder = default(output_folder, "outputs/simple_dub_sample/svd_xt/")
		# model_config = "scripts/sampling/configs/svd_xt.yaml"
	elif version == "svd_image_decoder":
		num_frames = default(num_frames, 14)
		num_steps = default(num_steps, 25)
		output_folder = default(output_folder, "outputs/simple_dub_sample/svd_image_decoder/")
		# model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
	elif version == "svd_xt_image_decoder":
		num_frames = default(num_frames, 25)
		num_steps = default(num_steps, 30)
		output_folder = default(output_folder, "outputs/simple_dub_sample/svd_xt_image_decoder/")
		# model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
	else:
		raise ValueError(f"Version {version} does not exist.")


	if use_latent:
		input_key = "latents"
	else:
		input_key = "frames"

	############# Load models #################
	model, filter, n_batch = load_model(
		model_config,
		device,
		num_frames,
		num_steps,
		input_key,
	)
	
	model.en_and_decode_n_samples_a_time = decoding_t
	if lora_path is not None:
		model.init_from_ckpt(lora_path, remove_keys_from_weights=None)
	torch.manual_seed(seed)

	print('\n')
	print('******* Save results in {} ********'.format(output_folder))
	
	audio_model = None
	if audio_path is not None and (audio_path.endswith(".wav") or audio_path.endswith(".mp3")):
		print('************* Load audio encoder: default is wav2vec2 *************')
		fps = fps_id + 1
		model_type = 'wav2vec2'
		model_size = 'base'
		audio_model = AudioWrapper(model_type=model_type, model_size=model_size, fps=fps)
		audio_model.eval()
		audio_model.cuda()

	audio, raw_audio = get_audio_embeddings(audio_path, 16000, fps_id + 1, audio_model = audio_model, save_emb = output_folder)
	
	video = read_video(video_path, output_format="TCHW")[0]
	video = (video / 255.0) * 2.0 - 1.0
	
	# max_seconds = 2
	if max_seconds is not None:
		max_frames = max_seconds * fps_id
		if video.shape[0] > max_frames:
			video = video[:max_frames]

	video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")    
	if os.path.exists(video_embedding_path):
		video_emb = None
		if use_latent:
			video_emb = load_safetensors(video_embedding_path)["latents"]
			print('video_emb', video_emb.shape)
	else:
		######### Load VAE ############# 
		print('************* load VAE *************')
		vae_model = VaeWrapper('video')
		video_emb = encode_video(video, vae_model, device)
		print('video_emb', video_emb.shape)

	
	
	if max_seconds is not None:
		max_frames = max_seconds * fps_id
		# if video.shape[0] > max_frames:
		audio = audio[:max_frames]
		video_emb = video_emb[:max_frames] if video_emb is not None else None
		raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
	audio = audio.cuda()

	
	h, w = video.shape[2:]
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
	
	# landmarks_path = video_path.replace(".mp4", ".npy")
	# if os.path.exists(landmarks_path):
	# 	landmarks = np.load(landmarks_path)
	# else:

	print('Extract landmarks')
	lmks_detector = face_alignment.FaceAlignment(
			face_alignment.LandmarksType.TWO_D,
			flip_input=False,
			device=device,
		)
	landmarks = get_landmarks_facealignment(model_input, lmks_detector)
	


	gt_chunks, masks_list, audio_list, emb_list, masks_big_list = create_interpolation_inputs(
		model_input, audio, landmarks, num_frames, video_emb, overlap, what_mask
	)
	masks_big = torch.stack(masks_big_list).to(device)
	gt_chunks = torch.stack(gt_chunks).to(device)

	check_masks(gt_chunks, masks_big, save_path = './masked_frame_co.png')
	
	gt_chunks = merge_overlapping_segments(gt_chunks, overlap)
	gt_chunks = torch.clamp((gt_chunks + 1.0) / 2.0, min=0.0, max=1.0)
	gt_vid = (gt_chunks * 255).cpu().numpy().astype(np.uint8)

	check_landmarks(gt_vid, landmarks, save_path = './land_co.png')
	
	
	# Take random index
	idx = torch.randint(0, len(model_input), (1,)).item()
	condition = model_input[idx].unsqueeze(0).to(device)
	condition_emb = video_emb[idx].unsqueeze(0).to(device) if video_emb is not None else None
	print('condition', condition.shape)
	print('condition_emb', condition_emb.shape)

	# for i in tqdm(range(len(masks_list)), desc="Autoregressive", total=len(masks_list)):
	masks = torch.stack(masks_list)
	print('masks', masks.shape)
	audio_cond = torch.stack(audio_list).to(device)
	embbedings = torch.stack(emb_list).to(device) if emb_list is not None else None
	print('embbedings', embbedings.shape)
	

	condition = repeat(condition, "b c h w -> (b d) c h w", d=audio_cond.shape[0])
	condition_emb = repeat(condition_emb, "b c h w -> (b d) c h w", d=audio_cond.shape[0])

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
	value_dict["masks"] = masks.transpose(1, 2).to(device)

	value_dict["cond_frames"] = condition_emb
	value_dict["cond_aug"] = cond_aug
	value_dict["audio_emb"] = audio_cond
	value_dict["gt"] = rearrange(embbedings, "b t c h w -> b c t h w").to(device)

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
				uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
				uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
				c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
				c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

			video = torch.randn(shape, device=device)

			n_batch *= embbedings.shape[0]

			additional_model_inputs = {}
			additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch, num_frames).to(device)
			additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

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

	os.makedirs(output_folder, exist_ok=True)
	# base_count = len(glob(os.path.join(output_folder, "*.mp4")))
	# video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
	# video_path_gt = os.path.join(output_folder, f"{base_count:06d}_gt.mp4")

	video_name = os.path.splitext(os.path.basename(video_path))[0]
	audio_name = os.path.splitext(os.path.basename(audio_path))[0]
	video_path = os.path.join(output_folder, '{}_audio_{}.mp4'.format(video_name, audio_name))
	video_path_gt = os.path.join(output_folder, '{}_audio_{}_gt.mp4'.format(video_name, audio_name))

	
	vid = (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
   
	if raw_audio is not None:
		raw_audio = rearrange(raw_audio[: vid.shape[0]], "f s -> () (f s)")

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
	print(f"Saved gt video to {video_path_gt}")


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
