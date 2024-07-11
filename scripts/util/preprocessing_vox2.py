"""
Crop frames for LSR:
VOxCeleb2: using the bounding boxes from metadata and scale_crop =1.0 
VoxCeleb1: the bounding boxes from metadata are not correct in order to crop images for LSR. Need to find a way to crop images similar to VoxCeleb2
Assuming that the mean bounding box from Vox2 is 224x224 and the mean bounding box from Vox1 is 156x156 ratio = 224/156 = 1.5 
find the optimal shift_x and shift_y for the new bounding box.
"""

import os
import numpy as np
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import torch

from facenet_pytorch import MTCNN
from scripts.util.utils_video import read_video
from scripts.util.utilities import save_image

REF_SIZE = 360 # Height
LOW_RES_SIZE = 400

def crop_video_vox2(video_path):
	mtcnn = MTCNN(keep_all=True, device='cpu')
	
	vr = read_video(video_path)
	full_vid_length = len(vr)
	end_frame = full_vid_length
	num_frames = 10; start_frame = 0
	if num_frames is not None:
		end_frame = min(start_frame + num_frames, full_vid_length)   
			
	numpy_video = vr.get_batch(range(start_frame, end_frame)).asnumpy()
	image_size = (512, 512)
	all_bboxes = []
	all_landmarks = []
	for rgb_frame in tqdm(numpy_video):
		boxes, _ = mtcnn.detect(rgb_frame)
		if boxes is None:
			all_bboxes.append([])
			all_landmarks.append([])
		else:
			# Process each face detected
			frame_landmarks = []
			frame_bboxes = []
			for box in boxes[:1]:
				# Convert box to dlib rectangle
				frame_bboxes.append([int(x) for x in box])
			all_bboxes.append(frame_bboxes)
			all_landmarks.append(frame_landmarks)
	
	frames_1 = torch.from_numpy(numpy_video).permute(0, 3, 1 ,2) # N x 3 x 256 x 256
	bboxes_1 = torch.from_numpy(np.concatenate(all_bboxes))

	for indx, (frame, bbox) in enumerate(zip(frames_1, bboxes_1)):
		print(bbox, frame.shape)
		x, y, xmax, ymax = bbox
		cropped = frame[:, x : x + xmax, y : y + ymax ]
		print(cropped.shape)
		save_image(cropped.float(), './cropped.png')
		quit()
		cropped_image, _ = image_resize(cropped_image, width = image_size[0], height = image_size[1])
		# image_name = image_file.split('/')[-1]
		# filename = os.path.join(save_dir, image_name)
		# cv2.imwrite(filename,  cv2.cvtColor(cropped_image.copy(), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
	
	video_path = '/fsx/behavioural_computing_data/face_generation_data/AA_processed/part_14/video_aligned_512/A089_C001_0802HK_001_output_output.mp4'
	crop_video_vox2(video_path)
