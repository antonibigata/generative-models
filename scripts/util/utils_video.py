import decord
import numpy as np
import torch
from torchvision import transforms
import subprocess
import cv2
from einops import rearrange, repeat
import math
from tqdm import tqdm

import scripts.libs.dtk.transforms as dtf
import scripts.libs.dtk.nn as dnn
import scripts.libs.dtk.media as dtm
from scripts.util.preprocessing import preprocess_AI_agents
from scripts.util.utils_audio import load_audio

def convert_video_fps(input_video, out_video, target_fps = 25):

	# audio = load_audio(input_video)
	# audio_path = 'test.wav'
	# dtm.save_audio(audio_path, audio, 16000)
	# audio_file_path = audio_path

	command = ' '.join([
		"ffmpeg", "-i", input_video,
		# "-i", audio_file_path,      # Input audio file
		"-strict", "-2", 			# Some legacy arguments
		"-c:a", "aac",              # AAC audio codec
		"-loglevel", "quiet",       # Verbosity arguments
		"-qscale", "0", 			# Preserve the quality
		"-r", f"{target_fps}", 				# save with 25 fps
		"-y", 						# Overwrite if the file exists
		out_video
	])

	return_code = subprocess.call(command, shell=True)
	success = (return_code == 0)
	return success


def get_video_fps(video_path, get_frames = True):
	vr = decord.VideoReader(video_path)
	fps = vr.get_avg_fps()
	full_vid_length = len(vr)

	native_height, native_width, channels = vr.next().shape
	return fps, full_vid_length, (native_height, native_width)

def read_video(video_path, retries=10):
	try:	
		return decord.VideoReader(
			video_path,
		)
	except Exception as e:
		if retries <= 0:
			raise e
		else:
			return read_video(video_path, retries=retries - 1)

def get_video_transform(norm_mean, norm_std):
	transform_chain = [dtf.ToTensorVideo()]
	transform_chain += [dtf.NormalizeVideo(norm_mean, norm_std)]
	video_transform = transforms.Compose(transform_chain)
	return video_transform

def preprocess_video(video_path, start_frame, image_size, norm_mean, norm_std, num_frames = None, align = False, scale_factor = 1.8):
		
	video_transform = get_video_transform(norm_mean, norm_std)
	# Create a tensor with the video
	vr = read_video(video_path)
	native_height, native_width, channels = vr.next().shape
	# print(native_height, native_width, channels)
	full_vid_length = len(vr)
	end_frame = full_vid_length
	if num_frames is not None:
		end_frame = min(start_frame + num_frames, full_vid_length)   
			
	numpy_video = vr.get_batch(range(start_frame, end_frame)).asnumpy()
	num_frames, or_h, or_w, c = numpy_video.shape
	original_video = numpy_video.copy()
	
	stats_alignment = {}; aligned_keypoints = None
	if align:
		numpy_video, aligned_keypoints, stats_alignment = preprocess_AI_agents(numpy_video, image_size = image_size, scale_factor = scale_factor)  # numpy video (N x H x W x 3) [0,255]  
		numpy_video = numpy_video.astype(np.uint8)		
	else:
		numpy_video_res = []
		for rgb_frame in (numpy_video):
			rgb_frame = cv2.resize(rgb_frame, image_size)	
			numpy_video_res.append(rgb_frame)
		numpy_video_res = np.asarray(numpy_video_res)
		numpy_video = numpy_video_res

	video_data = torch.from_numpy(numpy_video)
	video_data = video_transform(video_data)
	
	return video_data, start_frame, end_frame, full_vid_length, stats_alignment, original_video, aligned_keypoints

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

