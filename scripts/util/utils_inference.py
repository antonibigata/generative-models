import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import math

from sgm.data.data_utils import (
	create_masks_from_landmarks_full_size,
	create_face_mask_from_landmarks,
	create_masks_from_landmarks_box,
)

def get_unique_embedder_keys_from_conditioner(conditioner):
	return list(set([x.input_key for x in conditioner.embedders]))


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
	# print(video.shape)
	# Adjustment for overlap to ensure segments are created properly
	# print('num_frames {}, overlap {}'.format(num_frames, overlap))
	step = num_frames - overlap

	# Ensure there's at least one step forward on each iteration
	if step < 1:
		step = 1
	
	# TODO: Take the last frame if they are less than 14 frames 
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
	
	# if segment_end < video.shape[0]:
	# 	i = video.shape[0] - num_frames
	# 	segment_end = video.shape[0] 
	# 	if what_mask == "full":
	# 		masks = create_masks_from_landmarks_full_size(
	# 			landmarks[i:segment_end, :], video.shape[-1], video.shape[-2], offset=-0.01
	# 		)
	# 	elif what_mask == "box":
	# 		masks = create_masks_from_landmarks_box(landmarks[i:segment_end, :], (video.shape[-1], video.shape[-2]))
	# 	else:
	# 		masks = create_face_mask_from_landmarks(
	# 			landmarks[i:segment_end, :],
	# 			video.shape[-1],
	# 			video.shape[-2],
	# 			mask_expand=0.05,
	# 		)
	# 	masks_chunks_big.append(masks)
	# 	gt_chunks.append(video[i:segment_end])
	# 	masks = F.interpolate(masks.unsqueeze(1).float(), size=(64, 64), mode="nearest")
	# 	masks_chunks.append(masks)
	# 	if video_emb is not None:
	# 		video_emb_chunks.append(video_emb[i:segment_end])
	# 	if audio is not None:
	# 		audio_chunks.append(audio[i:segment_end])

	return gt_chunks, masks_chunks, audio_chunks, video_emb_chunks, masks_chunks_big

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
