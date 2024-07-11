import torchaudio
import torch
import math
from einops import rearrange, repeat
import os

from sgm.util import default, instantiate_from_config, trim_pad_audio, get_raw_audio, save_audio_video
from scripts.util.utilities import get_name_from_file

def load_audio(audio_path, rate = 16000):
	audio_data, sample_rate = torchaudio.load(audio_path, channels_first=True) # channels x samples
	if sample_rate != rate:
		# print('Resample from {} to {}'.format(sample_rate, rate))
		audio_data = torchaudio.functional.resample(audio_data, sample_rate, rate)
	if audio_data.size(0) > 1:
		# Convert to mono
		audio_data = audio_data[0].unsqueeze(0)
	audio_data = audio_data.T
	
	return audio_data


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
			raw_audio = trim_pad_audio(raw_audio, audio_rate, max_len_raw=math.ceil(n_frames) * samples_per_frame)
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

	elif audio_path is not None and audio_path.endswith(".pt"):
		audio_embeddings = torch.load(audio_path)
		raw_audio_path = audio_path.replace(".pt", ".wav").replace("_whisper_emb", "").replace("_wav2vec2_emb", "")

		if os.path.exists(raw_audio_path):
			raw_audio = get_raw_audio(raw_audio_path, audio_rate)
		else:
			print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")
	return audio_embeddings, raw_audio
