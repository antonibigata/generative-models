import os
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip, clips_array, TextClip, CompositeVideoClip

from scripts.util.utilities import make_path

def collect_matching_videos(folders):
	video_files = {}
	for folder in folders:
		for filename in os.listdir(folder):
			if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
				if filename not in video_files:
					video_files[filename] = []
				video_files[filename].append(os.path.join(folder, filename))

	print('Found {} matching video files'.format(len(video_files)))
	return video_files


def crop_half_width(clip):
	width, height = clip.size
	return clip.crop(x1=width-512, y1=0, x2=width, y2=height)

def concatenate_side_by_side(video_files, output_folder, folders, audio_index = 0):
	for filename, filepaths in tqdm(video_files.items()):
		if len(filepaths) == len(folders):   # Only concatenate if the matching file are in all folders 

			output_file = os.path.join(output_folder, filename)
			if os.path.exists(output_file):
				continue
			
			# video_clips = []
			# for i, path in enumerate(filepaths):
			# 	if i > 0:
			# 		video_clips.append(crop_half_width(VideoFileClip(path)))
			# 	else:
			# 		video_clips.append((VideoFileClip(path)))
			video_clips = [VideoFileClip(video) for video in filepaths]
			 
			min_height = min(clip.h for clip in video_clips)
			video_clips = [clip.resize(height=min_height) for clip in video_clips]

			min_duration = min(clip.duration for clip in video_clips)
			video_clips = [clip.subclip(0, min_duration) for clip in video_clips]

			final_clip = clips_array([video_clips])
		
			final_clip = final_clip.set_audio(video_clips[audio_index].audio)
			final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', logger=None)

def concatenate_videos(video_1, video_2, output_name):
	filepaths = [video_1, video_2]
	audio_index = 0
	video_clips = [VideoFileClip(video) for video in filepaths]
			 
	min_height = min(clip.h for clip in video_clips)
	video_clips = [clip.resize(height=min_height) for clip in video_clips]

	min_duration = min(clip.duration for clip in video_clips)
	video_clips = [clip.subclip(0, min_duration) for clip in video_clips]

	final_clip = clips_array([video_clips])

	final_clip = final_clip.set_audio(video_clips[audio_index].audio)
	final_clip.write_videofile(output_name, codec='libx264', audio_codec='aac', logger=None)

if __name__ == "__main__":
	
	# video_1 = '/data/home/stellab/projects/generative-models/outputs/simple_dub_sample/svd/A089_C001_0802HK_001_output_output_audio_RD_Radio18_000_wav2vec2_emb.mp4'
	# video_2 = '/data/home/stellab/projects/generative-models/outputs/test_artifacts/A089_C001_0802HK_001_output_output_25_audio_RD_Radio18_000_wav2vec2_emb.mp4'
	# concatenate_videos(video_1, video_2, '/data/home/stellab/projects/generative-models/outputs/test_artifacts/compare/difference_60_25fps.mp4')
	# quit()
	folder_1 = '/data/home/stellab/projects/generative-models/outputs/test_artifacts'
	folder_2 = '/data/home/stellab/projects/generative-models/outputs/test_artifacts_inference_code'

	folders = [folder_1, folder_2]
	video_files = collect_matching_videos(folders)
	if not video_files:
		print("No matching video files found in the provided folders.")
		exit()
	# output_path = './results_demo/test_set_actors/compare_ablation_wo_avsync'
	output_path = '/data/home/stellab/projects/generative-models/outputs/test_artifacts/compare'
	make_path(output_path)
	concatenate_side_by_side(video_files, output_path, folders, audio_index = 1)
