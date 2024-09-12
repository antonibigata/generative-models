import math
import os
from glob import glob
from pathlib import Path
from typing import Optional
from tqdm import tqdm  # Correct import
import time
import face_alignment
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
import torchaudio
from safetensors.torch import load_file as load_safetensors
from torchvision.io import read_video

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from sgm.util import default, instantiate_from_config, save_audio_video
from scripts.util.audio_wrapper import AudioWrapper
from scripts.util.vae_wrapper import VaeWrapper
from scripts.util.utilities import *
from scripts.util.utils_audio import get_audio_embeddings
from scripts.util.utils_inference import *
from scripts.util.utils_video import preprocess_video, get_video_fps, convert_video_fps, encode_video
from scripts.util.preprocessing import invert_transformation_ai_agents
"""
python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
 	--video_path=/fsx/rs2517/data/HDTF/cropped_videos_original/WDA_BarackObama_000.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/RD_Radio18_000_wav2vec2_emb.pt \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=5  --preprocess_video_flag False --blend_to_original False \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10 \
	--output_folder ./outputs/test_artifacts/

	/fsx/behavioural_computing_data/voxceleb2/test/id00017/01dfn2spqyE/00001.mp4
	/fsx/behavioural_computing_data/voxceleb2/test/id04030/7mXUMuo5_NE/00001.mp4
	/fsx/stellab/AI_agents/test_set_all_actors_25fps/A479_C022_1120YY_001.mp4
	/fsx/behavioural_computing_data/voxceleb2/test/id00562/0Zh6t-f8MsY/00001.mp4

	/fsx/behavioural_computing_data/voxceleb2/test/id04006/113VkmVVz1Q/ 
	
	part_27/video_aligned_512/A076_C034_0731PQ_001_output_output.mp4 -> poses
	/fsx/behavioural_computing_data/face_generation_data/AI_Agent_dataset/part_27

part_1/video_aligned_512/A662_C030_01290I_001_output_output.mp4
part_27/video_aligned_512/A057_C005_0725BR_001_output_output.mp4
part_1/video_aligned_512/A564_C034_1217I2_001_output_output.mp4
--video_path=/fsx/behavioural_computing_data/face_generation_data/AI_Agent_dataset/part_27/A076_C034_0731PQ_001.mov

part_14/video_aligned_512/A095_C027_0802M4_001_output_output.mp4

	/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4

	/fsx/behavioural_computing_data/face_generation_data/AA_processed/

	--video_path=/data/home/stellab/projects/dubbing_gans/results_demo/insta_5/3345252620443890114_0_chain0_translated.mp4
	--audio_path=/fsx/rs2517/data/HDTF/audio/RD_Radio18_000_wav2vec2_emb.pt \
	--audio_path=/data/home/stellab/projects/dubbing_gans/results_demo/evaluation_data/kimmel.wav \
	--audio_path=/fsx/rs2517/data/HDTF/audio/WDA_BarackObama_000_wav2vec2_emb.pt
	   --video_path=/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4 \
	 --video_path=./A043_C047_0721BS_001_output_output.mp4 \ WDA_HillaryClinton_000

python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/rs2517/data/HDTF/cropped_videos_original/WDA_BarackObama_000.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/WDA_BarackObama_000_wav2vec2_emb.pt \
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
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=jawline --overlap=5 --chunk_size=10 \
	--output_folder ./outputs/new_runs_sept/

python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_27/video_aligned_512/A043_C047_0721BS_001_output_output.mp4 \
	--audio_path=/fsx/behavioural_computing_data/face_generation_data/HDTF/audio/WRA_DeanHeller_000.wav \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=20 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=jawline --overlap=5 --chunk_size=10 \
	--output_folder ./outputs/new_runs_sept/

python scripts/sampling/simple_dubbing_multi_stella.py --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--video_path=/fsx/rs2517/data/HDTF/video_crop/WRA_MikeEnzi_000.mp4 \
	--audio_path=/fsx/rs2517/data/HDTF/audio/WDA_DebbieStabenow0_000.wav \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=20 \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=jawline --overlap=5 --chunk_size=10 \
	--output_folder ./outputs/new_runs_sept/

	HDTF:
	/fsx/rs2517/data/HDTF/video_crop/WDA_StenyHoyer_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_DebbieStabenow0_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WRA_MikeEnzi_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WRA_MittRomney_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_RobinKelly_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_BarackObama_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_BarackObama_001.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_DonnaShalala1_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_JohnYarmuth1_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_JamesClyburn1_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_ChrisCoons1_000.mp4
/fsx/rs2517/data/HDTF/video_crop/WDA_LucilleRoybal-Allard_000.mp4

""" 


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
	preprocess_video_flag: bool = False,  # Preprocess video using actors preprocessing in needed
	start_frame: int = 0,
	blend_to_original: bool = False,
	reference_index: int = None,
	use_current_reference: bool = False,
	save_gt_video: bool = False
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
	
	os.makedirs(output_folder, exist_ok=True)
	
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
	
	fps, num_video_frames, video_size = get_video_fps(video_path, get_frames = True)	
	print('Video frames ', num_video_frames)
	if fps != fps_id + 1:
		print('Convert video fps from {} to {}'.format(fps, fps_id+1))
		video_name = os.path.splitext(os.path.basename(video_path))[0]
		video_path_25 = os.path.join(output_folder, '{}_25.mp4'.format(video_name))
		success = convert_video_fps(video_path, video_path_25, target_fps = fps_id + 1)
		if not success:
			print('Failed to change the fps for input video')
			exit()
		video_path = video_path_25

	# max_seconds = 1
	if max_seconds is not None:
		num_video_frames = max_seconds * fps_id
	
	norm_mean = [0.5, 0.5, 0.5]; norm_std = [0.5, 0.5, 0.5]
	if preprocess_video_flag:
		print('Preprocess video')
		video, start_frame, end_frame, full_vid_length, stats_alignment, original_video, landmarks = preprocess_video(video_path, start_frame, (512, 512), norm_mean, norm_std, num_frames = num_video_frames, align = True)
	else:
		video, start_frame, end_frame, full_vid_length, stats_alignment, original_video, landmarks = preprocess_video(video_path, start_frame, (512, 512), norm_mean, norm_std, num_frames = num_video_frames, align = False)
		# video = read_video(video_path, output_format="TCHW")[0]
		# video = (video / 255.0) * 2.0 - 1.0
		
	# if max_seconds is not None:
	# 	max_frames = max_seconds * fps_id
	# 	if video.shape[0] > max_frames:
	# 		video = video[:max_frames]

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
		t0 = time.time()
		video_emb = encode_video(video, vae_model, device)
		print('video_emb', video_emb.shape)
		print('Time to encode video frames is {:.3f}'.format(time.time()-t0))

	if max_seconds is not None:
		max_frames = max_seconds * fps_id
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
	
	if landmarks is None:
		print('Extract landmarks')
		lmks_detector = face_alignment.FaceAlignment(
				face_alignment.LandmarksType.TWO_D,
				flip_input=False,
				device=device,
			)
		landmarks = get_landmarks_facealignment(model_input, lmks_detector)
	
	
	if model_input.shape[0] > audio.shape[0]:
		model_input = model_input[:audio.shape[0]]
	elif model_input.shape[0] < audio.shape[0]:
		audio = audio[:model_input.shape[0]]

	gt_chunks, masks_list, audio_list, emb_list, masks_big_list = create_interpolation_inputs(
		model_input, audio, landmarks, num_frames, video_emb, overlap, what_mask
	)

	masks_big = torch.stack(masks_big_list).to(device)
	gt_chunks = torch.stack(gt_chunks).to(device)
	
	# check_masks(gt_chunks, masks_big, save_path = './masked_frame_co.png')
	
	gt_chunks = merge_overlapping_segments(gt_chunks, overlap)
	gt_chunks = torch.clamp((gt_chunks + 1.0) / 2.0, min=0.0, max=1.0)
	gt_vid = (gt_chunks * 255).cpu().numpy().astype(np.uint8)

	# landmarks = data['landmarks'][0] * (512/224)
	# check_landmarks(gt_vid, landmarks, save_path = './land_co.png')
	

	# Take random index
	# TODO: Take the first frame of the 14 frames sequence and not a random one 
	if reference_index is None:
		idx = torch.randint(0, len(model_input), (1,)).item()
	else:
		idx = reference_index # Use the first reference frame
	condition = model_input[idx].unsqueeze(0).to(device)
	condition_emb = video_emb[idx].unsqueeze(0).to(device) if video_emb is not None else None
	print('condition', condition.shape)
	print('condition_emb', condition_emb.shape)
	print('model_input', model_input.shape)
	print('video_emb', video_emb.shape)

	masks = torch.stack(masks_list)
	print('masks', masks.shape)
	audio_cond = torch.stack(audio_list).to(device)
	embbedings = torch.stack(emb_list).to(device) if emb_list is not None else None
	print('video embbedings', embbedings.shape)
	print('audio_cond', audio_cond.shape)
	

	# condition torch.Size([1, 3, 512, 512])
	# condition_emb torch.Size([1, 4, 64, 64])
	# masks torch.Size([12, 14, 1, 64, 64])
	# embbedings torch.Size([12, 14, 4, 64, 64])
	# audio_cond torch.Size([12, 14, 2, 768])
	
	if not use_current_reference:
		condition = repeat(condition, "b c h w -> (b d) c h w", d=audio_cond.shape[0])
		condition_emb = repeat(condition_emb, "b c h w -> (b d) c h w", d=audio_cond.shape[0])
	else:
		# TODO: Check if for every batch i can use a difference reference frame
		condition = gt_chunks # Not correct
		condition_emb = embbedings
	print('condition', condition.shape)
	print('condition_emb', condition_emb.shape)
	# condition torch.Size([13, 3, 512, 512])
	# condition_emb torch.Size([13, 4, 64, 64])

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

	t0 = time.time()
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

			additional_model_inputs = {}
			additional_model_inputs["image_only_indicator"] = torch.zeros(n_batch, num_frames).to(device)
			# print('image_only_indicator', additional_model_inputs["image_only_indicator"].shape)
			
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
			print('samples_z', samples_z.shape)
			samples_x = model.decode_first_stage(samples_z)

			samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

			video = None
	
	print('Time to generate video of {} frames is {:.3f}'.format(num_video_frames, time.time()-t0))
	samples = rearrange(samples, "(b t) c h w -> b t c h w", t=num_frames)
	samples = merge_overlapping_segments(samples, overlap)

	
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

	if save_gt_video:
		save_audio_video(
			gt_vid,
			audio=raw_audio,
			frame_rate=fps_id + 1,
			sample_rate=16000,
			save_path=video_path_gt,
			keep_intermediate=False,
		)
		print(f"Saved gt video to {video_path_gt}")

	print(f"Saved video to {video_path}")
	
	if blend_to_original:
		vid = rearrange(vid, "t c h w -> t h w c")
		
		if original_video.shape[0] > vid.shape[0]:
			original_video = original_video[:vid.shape[0]]
		blended_vid = invert_transformation_ai_agents(original_video, vid, stats_alignment, lmks_detector = None, kernel_size = 50, blend_full_face = True, sigmoid = True)
		blended_vid = rearrange(blended_vid, "t h w c -> t c h w")
		video_path = os.path.join(output_folder, '{}_audio_{}_or.mp4'.format(video_name, audio_name))
		save_audio_video(
			blended_vid,
			audio=raw_audio,
			frame_rate=fps_id + 1,
			sample_rate=16000,
			save_path=video_path,
			keep_intermediate=False,
		)
		print(f"Saved video to {video_path}")


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
