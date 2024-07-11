import torch
from facenet_pytorch import MTCNN
import dlib
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
# from utilities.preprocessing_utils import *
# from utilities.utils import torch_video_numpy
# from utilities.mask import *
# from utilities.blending_code import blending_matrix, adjust_bbox_size

from scripts.util.preprocessing_utils import *
from scripts.util.blending_code import blending_matrix, adjust_bbox_size

def torch_video_numpy(video):
	video = video.permute(0, 2, 3, 1)
	video = video.detach().cpu().numpy()
	min_val = -1; max_val = 1
	video = (video - min_val)/(max_val - min_val + 1e-5)
	video = video * 255
	video = video.astype(np.uint8)
	return video

def align_one_video(
	frames: torch.Tensor,
	bboxes: torch.Tensor,
	keypoints: torch.Tensor,
	scale_factor: float = 1.8,
	output_size_align: Tuple[int, int] = (512, 512),
	window_size: int = 7,
	max_sigma: float = 8,
	min_sigma: float = 2,
	threshold: float = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:

	stats_alignment = {}
	frames = frames.permute(0, 2, 3, 1)
	aligned_frames = []
	aligned_keypoints = []

	initial_bbox = adjust_bbox(
		bboxes[0], scale_factor, frames[0].shape[1], frames[0].shape[0]
	)
	fixed_size = min(initial_bbox[2], initial_bbox[3])

	raw_bboxes = []
	for face in bboxes:
		x, y, xmax, ymax = face
		w = xmax - x
		h = ymax - y
		center_x = x + w // 2
		center_y = y + h // 2
		new_x = center_x - fixed_size // 2
		new_y = center_y - fixed_size // 2
		raw_bboxes.append((new_x, new_y, fixed_size, fixed_size))

	stabilized_bboxes = stabilize_bbox_eg(
		raw_bboxes, window_size, max_sigma, min_sigma, threshold
	)

	cropped_areas = []
	for indx, (frame, bbox) in enumerate(zip(frames, stabilized_bboxes)):
		x, y, w, h = bbox
		
		padded_frame = np.zeros(
			(
				frame.shape[0] + 2 * fixed_size,
				frame.shape[1] + 2 * fixed_size,
				frame.shape[2],
			),
			dtype=np.uint8,
		)
		padded_frame[
			fixed_size : fixed_size + frame.shape[0],
			fixed_size : fixed_size + frame.shape[1],
		] = frame
		x_padded, y_padded = x + fixed_size, y + fixed_size
		cropped = padded_frame[y_padded : y_padded + h, x_padded : x_padded + w]
		cropped_areas.append([y_padded, y_padded +h, x_padded, x_padded + w])

		original_size = cropped.shape[:2]
		resized = cv2.resize(cropped, output_size_align)
		aligned_frames.append(resized)

		# Adjust keypoints boxes
		scale_x, scale_y = output_size_align[0] / w, output_size_align[1] / h
		frame_keypoints = keypoints[indx]
		adjusted_keypoints = []
		for kx, ky in frame_keypoints:
			adjusted_kx = (kx - x) * scale_x
			adjusted_ky = (ky - y) * scale_y
			adjusted_keypoints.append((adjusted_kx, adjusted_ky))
		aligned_keypoints.append(adjusted_keypoints)

	stats_alignment = {
		'original_size': 	original_size,
		'cropped_areas':	cropped_areas,
		'fixed_size':		fixed_size
	}	

	
	return np.stack(aligned_frames), np.asarray(aligned_keypoints), stats_alignment

" Preprocess video frames similar to AI agents preprocessing pipeline"
def preprocess_AI_agents(numpy_video, image_size = (256, 256), scale_factor = 1.8):
	mtcnn = MTCNN(keep_all=True, device='cpu')

	predictor_path = "/fsx/nikitadrobyshev/diffusion/data/shape_predictor_68_face_landmarks.dat"
	predictor = dlib.shape_predictor(predictor_path)

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
				rect = dlib.rectangle(left=int(box[0]), top=int(box[1]), right=int(box[2]), bottom=int(box[3]))
				frame_bboxes.append([int(x) for x in box])

				# Get landmarks
				shape = predictor(rgb_frame, rect)
				landmarks = [(shape.part(n).x, shape.part(n).y) for n in range(68)]
				frame_landmarks.append(landmarks)

			all_bboxes.append(frame_bboxes)
			all_landmarks.append(frame_landmarks)
	
	frames_1 = torch.from_numpy(numpy_video).permute(0, 3, 1 ,2) # N x 3 x 256 x 256
	bboxes_1 = torch.from_numpy(np.concatenate(all_bboxes))
	landmarks_1 = torch.from_numpy(np.concatenate(all_landmarks))

	aligned_frames, aligned_keypoints, stats_alignment = align_one_video(frames_1, bboxes_1, landmarks_1, scale_factor = scale_factor, output_size_align = image_size)
	
	return aligned_frames, aligned_keypoints, stats_alignment

def get_unaligned_frame(original_frame, aligned_frame, fixed_size, original_size, cropped_area):
	padded_frame = np.zeros(
		(
			original_frame.shape[0] + 2 * fixed_size,
			original_frame.shape[1] + 2 * fixed_size,
			original_frame.shape[2],
		),
		dtype=np.uint8,
	)
	aligned_frame = cv2.resize(aligned_frame, original_size)
	a = cropped_area[0]; b = cropped_area[1]; c = cropped_area[2]; d = cropped_area[3]
	padded_frame[a : b, c : d, : ] = aligned_frame
	unaligned_frame = padded_frame[
		fixed_size : fixed_size + original_frame.shape[0],
		fixed_size : fixed_size + original_frame.shape[1],
	]	

	return unaligned_frame, a, b, c, d

def get_landmarks(frame, lmks_detector):
	landmarks = lmks_detector.get_landmarks(frame)
	if landmarks is None:
		return None
	landmarks = landmarks[0]
	return landmarks

def invert_transformation_ai_agents(original_frames, video_gen, stats_alignment, lmks_detector = None, kernel_size = 10, blend_full_face = True, sigmoid = False):
	
	# video_gen = torch_video_numpy(video_gen)
	original_size = stats_alignment['original_size']
	fixed_size = stats_alignment['fixed_size']
	cropped_areas = stats_alignment['cropped_areas']
	original_video = original_frames
	for indx, original_frame in enumerate(tqdm(original_frames)):
		cropped_area = cropped_areas[indx]
		aligned_frame = video_gen[indx]
		unaligned_frame, a, b, c, d = get_unaligned_frame(original_frame, aligned_frame, fixed_size, original_size, cropped_area)

		if blend_full_face:
			a_hat = a - fixed_size; b_hat = b - fixed_size
			c_hat = c - fixed_size; d_hat = d - fixed_size
			crop_left_x = c_hat ; crop_right_x = d_hat	
			crop_top_y = a_hat; crop_bottom_y = b_hat
			cropped_frame = unaligned_frame[crop_top_y:crop_bottom_y, crop_left_x:crop_right_x, :]
		
			if not sigmoid:
				original_video[indx, crop_top_y:crop_bottom_y, crop_left_x:crop_right_x] =  cropped_frame 
			else:
				sigmoid_steepness = -20; sigmoid_displacement = 150
				frame_1 = unaligned_frame
				frame_2 = original_frame			
				blended_frame = sigmoid_blending(frame_1, frame_2, crop_top_y ,crop_bottom_y, crop_right_x, crop_left_x, 
						sigmoid_steepness = sigmoid_steepness, sigmoid_displacement = sigmoid_displacement)
				original_video[indx] = blended_frame
		# else:	
		# 	landmarks_gen = get_landmarks(unaligned_frame, lmks_detector)
		# 	landmarks_or = get_landmarks(original_frame, lmks_detector)
		# 	if  landmarks_or[8, 1] > landmarks_gen[8, 1]:
		# 		landmarks_gen = landmarks_or
				
		# 	mask, mask_box = face_mask_jaw_box(unaligned_frame.shape[:2], landmarks_gen, kernel_size = kernel_size)
		# 	mask = (1-mask) * 255
		# 	crop_left_x =  mask_box[0][0]; crop_right_x =  mask_box[3][0]  
		# 	crop_top_y =  mask_box[0][1]; crop_bottom_y =  mask_box[2][1]
		# 	sigmoid_steepness = -20.0; sigmoid_displacement = 150.0
		
		# 	frame_1 = unaligned_frame
		# 	frame_2 = original_frame
		# 	unaligned_frame = restore_img(frame_2, frame_1, mask)
		# 	unaligned_frame = sigmoid_blending(unaligned_frame, frame_2, crop_top_y ,crop_bottom_y, crop_right_x, crop_left_x, 
		# 		sigmoid_steepness = sigmoid_steepness, sigmoid_displacement = sigmoid_displacement)
		# 	original_video[indx] = unaligned_frame
		
	return original_video

def sigmoid_blending(frame_1, frame_2, crop_top_y ,crop_bottom_y, crop_right_x, crop_left_x, sigmoid_steepness = -20, sigmoid_displacement = 150):
	t_margin = crop_top_y; b_margin = crop_bottom_y
	r_margin = crop_right_x; l_margin = crop_left_x
	
	if int(b_margin - t_margin) % 2 != 0:
		t_margin = t_margin+1
	if int(r_margin - l_margin) % 2 != 0:
		r_margin = r_margin+1

	# ensure that box is inside the image dimensions
	if t_margin < 0:
		t_margin = 0
	if l_margin < 0:
		l_margin = 0
	
	ff = frame_2.copy()
	
	# displacement -> more 0s at the border i.e., more real frame
	blending_m = blending_matrix(
		h=int(b_margin - t_margin),
		w=int(r_margin - l_margin),
		sigmoid_steepness=sigmoid_steepness,
		sigmoid_displacement=sigmoid_displacement,
	)    
	bb_t = torch.from_numpy(blending_m).unsqueeze(-1).repeat(1, 1, 3).numpy()
	expr_sq = ff[
		t_margin:b_margin,
		l_margin:r_margin,
	]
	speaking_sq = frame_1[
		t_margin:b_margin,
		l_margin:r_margin,
	]   
	# ensure that all have the same dimensions
	if not (speaking_sq.shape == bb_t.shape and bb_t.shape == expr_sq.shape):
		bb_t = adjust_bbox_size(bb_t, speaking_sq, expr_sq)                
	blend = bb_t * speaking_sq + (1 - bb_t) * expr_sq
	ff[
		t_margin:b_margin,
		l_margin:r_margin,
	] = blend

	return ff


def restore_img(input_img, generated_face, mask, use_color_norm = False):
	upscale_factor = 1
	
	inv_mask_erosion = cv2.erode(mask, np.ones((int(2 * upscale_factor), int(2 * upscale_factor)), np.uint8))
	inv_mask_erosion = inv_mask_erosion / 255
	pasted_face = inv_mask_erosion[:, :, None] * generated_face
	
	total_face_area = np.sum(inv_mask_erosion)
	w_edge = int(total_face_area**0.5) // 20
	erosion_radius = w_edge * 2
	inv_mask_erosion = inv_mask_erosion * 255
	inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
	blur_size = w_edge  * 2
	inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
	inv_soft_mask = inv_soft_mask[:, :, None]
	
	inv_soft_mask = inv_soft_mask / 255
	inv_soft_mask = np.repeat(inv_soft_mask, 3, axis=2)
	pasted_face = pasted_face.astype(np.uint8)

	########################################################
	# Normalize pasted frame -> it works only for instagram reels 
	if use_color_norm:
		pasted_face = (pasted_face - np.min(pasted_face)) / (np.max(pasted_face) - np.min(pasted_face))
		pasted_face = (pasted_face * 255).astype(np.uint8)
	########################################################

	input_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * input_img
	input_img = input_img.astype(np.uint8)
	return input_img