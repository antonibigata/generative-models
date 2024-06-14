import os
import torch
from torchvision import utils as torch_utils
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

def save_image(image, save_image_path):

	grid = torch_utils.save_image(
		image,
		save_image_path,
		normalize=True,
	)

def draw_landmarks(image, landmarks, save_path = None):

	if torch.is_tensor(image):
		# Convert image tensor to numpy array and transpose dimensions
		min_val = -1
		max_val = 1
		image.clamp_(min=min_val, max=max_val)
		image.add_(-min_val).div_(max_val - min_val + 1e-5)
		image = image #.mul(255.0).add(0.0) 
		image = image.cpu().numpy().transpose((1, 2, 0))
	

	# Plot image
	plt.imshow(image)
	# Plot landmarks
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
	plt.axis('off')

	if save_path is None:
		plt.savefig('lands.png', bbox_inches='tight', pad_inches=0)
	else:
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def make_path(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def get_name_from_file(file_path):
	base_filename = os.path.splitext(os.path.basename(file_path))[0]
	return base_filename


def get_landmarks_facealignment(video_frames, lmks_detector = None):
	assert lmks_detector is not None, "Error: lmks_detector should not be None"
	T, C, H, W = video_frames.shape
	landmarks_list = []
	for i in tqdm(range(T)):
		img_tmp = video_frames[i].clamp(-1, 1).add(1.0).div(2.0).mul(255.0)
		img_tmp = img_tmp.cpu().numpy()
		img_tmp = img_tmp.transpose(1, 2, 0)  # 256 x 256 x 3
		land = lmks_detector.get_landmarks(img_tmp)[0]
		landmarks_list.append(land)
	# landmarks_list = torch.from_numpy(np.asarray(landmarks_list))
	return np.asarray(landmarks_list)

def check_masks(gt_chunks, masks_big, save_path = './masked_frame_co.png'):
	ones = torch.ones(masks_big[0][0].shape) - 2  # pixels between [-1, 1]
	mask_tmp = 1 - masks_big[0][0]
	masked_frame = gt_chunks[0][0] * mask_tmp + mask_tmp + ones.to(gt_chunks.device)
	save_image(masked_frame, save_path)
	
 
def check_landmarks(gt_vid, landmarks, save_path = './land_co.png'):
	gt_vid_tmp = gt_vid.copy()
	gt_vid_tmp = rearrange(gt_vid_tmp, "t c h w -> t h w c")
	draw_landmarks(gt_vid_tmp[0], landmarks[0], save_path = save_path)
	