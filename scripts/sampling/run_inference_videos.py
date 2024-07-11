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
import torch.nn.functional as F
from torchvision.io import read_video
import random
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from sgm.util import default, instantiate_from_config, save_audio_video

from scripts.util.audio_wrapper import AudioWrapper
from scripts.util.vae_wrapper import VaeWrapper
from scripts.util.utilities import *
from scripts.util.utils_video import preprocess_video, get_video_fps, convert_video_fps, encode_video
from scripts.util.preprocessing import invert_transformation_ai_agents
from scripts.util.utils_audio import get_audio_embeddings
from scripts.util.utils_inference import *




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

def get_videos(test_set):
	video_names_save = None
	if test_set == 'actors':
		validation_path = '/fsx/stellab/AI_agents/test_set_actors_25fps' # videos_val_25 test_set_actors_25fps
		validation_txt = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/file_list_test_internal.txt'
		audio_path = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/'
	elif test_set == 'insta':
		validation_path = '/data/home/stellab/projects/dubbing_gans/results_demo/insta_5' # '/fsx/stellab/ig_test_set_25fps'
		videos_path = glob.glob(os.path.join(validation_path, '*.mp4'))
		videos_path.sort()
	elif test_set == 'vox':
		validation_txt = './file_list_val_vox.txt'
	elif test_set == 'hdtf':
		validation_txt = './file_list_val_hdtf.txt'
	
	audio_paths = []
	if test_set == 'actors':
		if validation_txt is not None:
			validation_path = '/fsx/stellab/AI_agents/test_set_all_actors_25fps'
			videos_path = []; unique_values = []; audio_paths = []
			with open(validation_txt, "r") as file:
				for line in file.readlines():
					file_name, video_ext = os.path.splitext(
						os.path.basename(line.rstrip())
					)
					line = line.rstrip()
					part_name = line.split('/')[0]

					file_name = file_name.split('_output_output')[0]
					id_name = file_name.split('_')[0]

					if id_name not in unique_values:
						unique_values.append(id_name)
						video_25_path = os.path.join(validation_path, file_name + '.mp4')
						videos_path.append(video_25_path)
						audio_paths.append(os.path.join(audio_path, part_name, 'audio', '{}.wav'.format(file_name)))

	elif test_set == 'vox':
		if validation_txt is not None:
			videos_path = []; unique_values = []; audio_paths = []; video_names_save = []
			validation_path = '/fsx/behavioural_computing_data/voxceleb2/test/'
			with open(validation_txt, "r") as file:
				for line in file.readlines():
					file_name, video_ext = os.path.splitext(
						os.path.basename(line.rstrip())
					)
					line = line.rstrip()				
					id_name = line.split('/')[0]
					video_id = line.split('/')[1]		

					if id_name not in unique_values:
						unique_values.append(id_name)
						video_25_path = os.path.join(validation_path, line)
						videos_path.append(video_25_path)
						video_names_save.append('{}_{}.mp4'.format(id_name, video_id))
		
		# get audio from actors 
		# validation_path = '/fsx/stellab/AI_agents/test_set_all_actors_25fps'
		# validation_txt = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/file_list_test_internal.txt'
		# audio_path = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/'
		# with open(validation_txt, "r") as file:
		# 	for line in file.readlines():
		# 		file_name, video_ext = os.path.splitext(
		# 			os.path.basename(line.rstrip())
		# 		)
		# 		line = line.rstrip()
		# 		part_name = line.split('/')[0]

		# 		file_name = file_name.split('_output_output')[0]
		# 		id_name = file_name.split('_')[0]
		# 		audio_paths.append(os.path.join(audio_path, part_name, 'audio', '{}.wav'.format(file_name)))

		validation_path = '/fsx/rs2517/data/HDTF/cropped_videos_original/'
		audio_path = '/fsx/behavioural_computing_data/face_generation_data/HDTF/audio'
		with open(validation_txt, "r") as file:
			for line in file.readlines():
				file_name, video_ext = os.path.splitext(
					os.path.basename(line.rstrip())
				)
				line = line.rstrip()
				video_name = line.split('.mp4')[0]
				audio_paths.append(os.path.join(audio_path, '{}.wav'.format(video_name)))
		random.shuffle(audio_paths)
		audio_paths = audio_paths[:len(videos_path)]
	
	elif test_set == 'hdtf':
		if validation_txt is not None:
			videos_path = []; unique_values = []; audio_paths = []
			validation_path = '/fsx/rs2517/data/HDTF/cropped_videos_original/'
			audio_path = '/fsx/behavioural_computing_data/face_generation_data/HDTF/audio'
			with open(validation_txt, "r") as file:
				for line in file.readlines():
					file_name, video_ext = os.path.splitext(
						os.path.basename(line.rstrip())
					)
					line = line.rstrip()
					video_name = line.split('.mp4')[0]
					unique_values.append(video_name)
					video_path = os.path.join(validation_path, line)
					videos_path.append(video_path)
					audio_paths.append(os.path.join(audio_path, '{}.wav'.format(video_name)))
		random.shuffle(audio_paths)
	return videos_path, audio_paths, video_names_save

def run_dubbing(model, model_input, audio, landmarks, num_frames, video_emb, overlap, what_mask, device, motion_bucket_id, 
				fps_id, cond_aug, num_video_frames, output_folder, blend_to_original, force_uc_zero_embeddings, n_batch,
				strength, stats_alignment, video_name, chunk_size, raw_audio, original_video):
	
	gt_chunks, masks_list, audio_list, emb_list, masks_big_list = create_interpolation_inputs(
		model_input, audio, landmarks, num_frames, video_emb, overlap, what_mask
	)

	masks_big = torch.stack(masks_big_list).to(device)
	gt_chunks = torch.stack(gt_chunks).to(device)
	# check_masks(gt_chunks, masks_big, save_path = './masked_frame_co.png')
	
	gt_chunks = merge_overlapping_segments(gt_chunks, overlap)
	gt_chunks = torch.clamp((gt_chunks + 1.0) / 2.0, min=0.0, max=1.0)
	gt_vid = (gt_chunks * 255).cpu().numpy().astype(np.uint8)
	# check_landmarks(gt_vid, landmarks, save_path = './land_co.png')
	
	# Take random index
	idx = torch.randint(0, len(model_input), (1,)).item()
	condition = model_input[idx].unsqueeze(0).to(device)
	condition_emb = video_emb[idx].unsqueeze(0).to(device) if video_emb is not None else None
	# print('condition', condition.shape)
	# print('condition_emb', condition_emb.shape)

	# for i in tqdm(range(len(masks_list)), desc="Autoregressive", total=len(masks_list)):
	masks = torch.stack(masks_list)
	# print('masks', masks.shape)
	audio_cond = torch.stack(audio_list).to(device)
	embbedings = torch.stack(emb_list).to(device) if emb_list is not None else None
	# print('embbedings', embbedings.shape)
	

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
			
			# print(video.shape, c.keys())
			# quit()
			samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)
			samples_x = model.decode_first_stage(samples_z)

			samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

			video = None
	
	print('Time to generate video of {} frames is {:.3f}'.format(num_video_frames, time.time()-t0))
	samples = rearrange(samples, "(b t) c h w -> b t c h w", t=num_frames)
	samples = merge_overlapping_segments(samples, overlap)

	
	video_path = os.path.join(output_folder, video_name)
	
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

	print(f"Saved video to {video_path}")
	if blend_to_original:
		vid = rearrange(vid, "t c h w -> t h w c")
		
		if original_video.shape[0] > vid.shape[0]:
			original_video = original_video[:vid.shape[0]]
		blended_vid = invert_transformation_ai_agents(original_video, vid, stats_alignment, lmks_detector = None, kernel_size = 50, blend_full_face = True, sigmoid = True)
		blended_vid = rearrange(blended_vid, "t h w c -> t c h w")
		video_path = os.path.join(output_folder, '{}_blend.mp4'.format(video_name))
		save_audio_video(
			blended_vid,
			audio=raw_audio,
			frame_rate=fps_id + 1,
			sample_rate=16000,
			save_path=video_path,
			keep_intermediate=False,
		)
		print(f"Saved video to {video_path}")


def main(
	test_set: str = 'actors',
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
	start_audio: int = 0,
	lora_path: Optional[str] = None,  # Not needed
	force_uc_zero_embeddings=[
		"cond_frames",
		"cond_frames_without_noise",
	],  # Useful for the classifier free guidance. What should be zeroed out in the unconditional embeddings
	chunk_size: int = None,  # Useful if the model gets OOM
	preprocess_video_flag: bool = False,  # Preprocess video using actors preprocessing in needed
	start_frame: int = 0,
	blend_to_original: bool = False,
	):

	"""
	Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
	image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
	"""
	
	num_frames = default(num_frames, 14)
	num_steps = default(num_steps, 25)
	# output_folder = default(output_folder, "outputs/simple_dub_sample/svd/")
	print('******* Save results in {} ********'.format(output_folder))
	os.makedirs(output_folder, exist_ok=True)

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

	print('************* Load audio encoder: default is wav2vec2 *************')
	fps = fps_id + 1
	model_type = 'wav2vec2'
	model_size = 'base'
	audio_model = AudioWrapper(model_type=model_type, model_size=model_size, fps=fps)
	audio_model.eval()
	audio_model.cuda()

	print('************* load VAE *************')
	vae_model = VaeWrapper('video')

	videos_path, audio_paths, video_names_save = get_videos(test_set)
	
	
	num_samples = len(videos_path)

	for i in range(num_samples):
		
		video_path = videos_path[i]
		# video_path = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4'
		if video_names_save is not None:
			out_video_name = video_names_save[i]
		else:
			out_video_name = '{}.mp4'.format(get_name_from_file(video_path))
		
		if os.path.exists(os.path.join(output_folder, out_video_name)):
			continue

		audio_path = audio_paths[i]
		audio_path = '/fsx/rs2517/data/HDTF/audio/RD_Radio18_000_wav2vec2_emb.pt' #'/fsx/behavioural_computing_data/face_generation_data/HDTF/audio/RD_Radio18_000.wav'

		fps, num_video_frames, video_size = get_video_fps(video_path, get_frames = True)	
		if max_seconds is not None:
			num_video_frames = max_seconds * fps_id

		print('{}/{}: Run video {}. Video frames {}, fps {}'.format(i, num_samples, out_video_name, num_video_frames, fps))
		if fps != fps_id + 1:
			print('Convert video fps from {} to {}'.format(fps, fps_id+1))
			video_name = os.path.splitext(os.path.basename(video_path))[0]
			video_path_25 = os.path.join(output_folder, '{}_25.mp4'.format(video_name))
			success = convert_video_fps(video_path, video_path_25, target_fps = fps_id + 1)
			if not success:
				print('Failed to change the fps for input video')
				exit()
			video_path = video_path_25
		
		
		audio, raw_audio = get_audio_embeddings(audio_path, 16000, fps_id + 1, audio_model = audio_model, save_emb = None)
		norm_mean = [0.5, 0.5, 0.5]; norm_std = [0.5, 0.5, 0.5]
		if preprocess_video_flag:
			print('Preprocess video')
			video, start_frame, end_frame, full_vid_length, stats_alignment, original_video, landmarks = preprocess_video(video_path, start_frame, (512, 512), norm_mean, 
																		norm_std, num_frames = num_video_frames, align = True, scale_factor= 1.6)
		else:
			video, start_frame, end_frame, full_vid_length, stats_alignment, original_video, landmarks = preprocess_video(video_path, start_frame, (512, 512), norm_mean, norm_std, num_frames = num_video_frames, align = False)

		num_video_frames = video.shape[0]
		video_emb = encode_video(video, vae_model, device)

		if max_seconds is not None:
			
			if start_audio != 0:
				start_audio_frames = start_audio * fps_id
				
				if start_audio_frames+num_video_frames > audio.shape[0]:
					print('Audio duration is not enought')
				audio = audio[start_audio_frames:start_audio_frames+num_video_frames]
				raw_audio = raw_audio[start_audio_frames:start_audio_frames+num_video_frames] if raw_audio is not None else None
			else:
				audio = audio[:num_video_frames]
				raw_audio = raw_audio[:num_video_frames] if raw_audio is not None else None
			video_emb = video_emb[:num_video_frames] if video_emb is not None else None
			

		audio = audio.cuda()
		model_input = video.cuda()
		
		if landmarks is None:
			print('Extract landmarks')
			lmks_detector = face_alignment.FaceAlignment(
					face_alignment.LandmarksType.TWO_D,
					flip_input=False,
					device=device,
				)
			landmarks = get_landmarks_facealignment(model_input, lmks_detector)

		run_dubbing(model, model_input, audio, landmarks, num_frames, video_emb, overlap, what_mask, device, motion_bucket_id, 
				fps_id, cond_aug, num_video_frames, output_folder, blend_to_original, force_uc_zero_embeddings, n_batch,
				strength, stats_alignment, out_video_name, chunk_size, raw_audio, original_video)

if __name__ == "__main__":
	Fire(main)
	"""
	python scripts/sampling/run_inference_videos.py --output_folder ./outputs/vox_new_box_2 --test_set hdtf --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=20 --start_audio=0  --preprocess_video_flag True --blend_to_original False \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10


	python scripts/sampling/run_inference_videos.py --test_set vox --output_folder ./outputs/voxceleb_box_2 --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=20  --preprocess_video_flag False --blend_to_original False \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10

	python scripts/sampling/run_inference_videos.py --test_set hdtf --output_folder ./outputs/hdtf_box_2 --fps_id 24 --motion_bucket_id 60 --cond_aug 0. --decoding_t 14 \
	--model_config=scripts/sampling/configs/svd_dub.yaml --max_seconds=20  --preprocess_video_flag False --blend_to_original False \
	--resize_size=512 --use_latent=True --num_steps=10 '--force_uc_zero_embeddings=[audio_emb]' --what_mask=box --overlap=5 --chunk_size=10

	"""
	