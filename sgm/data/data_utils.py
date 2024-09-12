import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from functools import partial
import math

" Used for half face "
def bbox2mask(img_shape, bbox, dtype='uint8'):
	"""Generate mask in ndarray from bbox.

	The returned mask has the shape of (h, w, 1). '1' indicates the
	hole and '0' indicates the valid regions.

	We prefer to use `uint8` as the data type of masks, which may be different
	from other codes in the community.

	Args:
		img_shape (tuple[int]): The size of the image.
		bbox (tuple[int]): Configuration tuple, (top, left, height, width)
		dtype (str): Indicate the data type of returned masks. Default: 'uint8'

	Return:
		numpy.ndarray: Mask in the shape of (h, w, 1).
	"""
	height, width = img_shape[:2]
	mask = np.ones((height, width, 1), dtype=dtype)
	# mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 0.0
	mask[bbox[0]:bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 0.0

	return mask


def create_masks_half_face(img_shape, num_frames):
	"""
	Create a mask that covers the lower half face without using landmarks
	"""
	h, w = img_shape[:2]
	mask = bbox2mask((h, w), (h // 2 - 20 , 0, h , w))
	mask = torch.tensor(np.transpose(mask, (2, 0, 1))).unsqueeze(0)
	mask = mask.repeat(num_frames, 1, 1, 1)
	mask = 1 - mask
	return mask

def create_masks_from_landmarks_box(landmark_list, img_shape, nose_index=33, dtype="uint8", box_expand=0.0):
	height, width = img_shape[:2]
	num_frames = landmark_list.shape[0]

	# Initialize the masks array
	masks = np.zeros((num_frames, height, width), dtype=dtype)

	if 0 <= box_expand < 1:
		box_expand = int(box_expand * width)

	for i in range(num_frames):
		# Get the landmarks for the current frame
		landmarks = landmark_list[i]

		# Get the y-coordinate of the nose landmark
		nose_point_h = landmarks[nose_index, 1]
		cut_h = nose_point_h

		# Find the leftmost and rightmost landmarks
		far_left_index = np.argmin(landmarks[:, 0])
		far_right_index = np.argmax(landmarks[:, 0])

		# Define the points for the mask contour
		left_up_point = np.array([landmarks[far_left_index][0] - box_expand, cut_h], dtype=np.int32)
		left_down_point = np.array([landmarks[far_left_index][0] - box_expand, height], dtype=np.int32)
		right_up_point = np.array([landmarks[far_right_index][0] + box_expand, cut_h], dtype=np.int32)
		right_down_point = np.array([landmarks[far_right_index][0] + box_expand, height], dtype=np.int32)

		# Define the contour
		contour = np.array([[left_up_point, left_down_point, right_down_point, right_up_point]])

		# Draw the contour on the mask
		cv2.drawContours(masks[i], [contour], -1, color=(1), thickness=cv2.FILLED)

	return torch.from_numpy(masks)


def create_masks_from_landmarks_cheeks(landmark_list, img_shape, dtype="uint8", box_expand=0.0):
	height, width = img_shape[:2]
	num_frames = landmark_list.shape[0]

	# Initialize the masks array
	masks = [] #np.zeros((num_frames, height, width), dtype=dtype)

	middle_nose_point = 29
	nose_index = 33

	if 0 <= box_expand < 1:
		box_expand = int(box_expand * width)

	for i in range(num_frames):
		# Get the landmarks for the current frame
		landmarks = landmark_list[i]

		mask_tmp = np.ones((height, width, 1), dtype=dtype)

		# Get the y-coordinate of the nose landmark
		cut_h = int(landmarks[middle_nose_point,1] )
		
		# Find the leftmost and rightmost landmarks
		far_left_index = np.argmin(landmarks[:, 0])
		far_right_index = np.argmax(landmarks[:, 0])

		# Define the points for the mask contour
		left_up_point = np.array([landmarks[far_left_index][0] - box_expand, cut_h], dtype=np.int32)
		left_down_point = np.array([landmarks[far_left_index][0] - box_expand, height], dtype=np.int32)
		right_up_point = np.array([landmarks[far_right_index][0] + box_expand, cut_h], dtype=np.int32)
		right_down_point = np.array([landmarks[far_right_index][0] + box_expand, height], dtype=np.int32)

		# Define the contour
		contour = np.array([[left_up_point, left_down_point, right_down_point, right_up_point]])

		# Draw the contour on the mask
		cv2.drawContours(mask_tmp, [contour], -1, color=(0), thickness=cv2.FILLED)

		#  Calculate the bounding box coordinates for the nose
		nose_right_dist = (landmarks[middle_nose_point][0] - landmarks[1][0]) * 0.10
		nose_left_dist = (landmarks[15][0] - landmarks[middle_nose_point][0]) * 0.10
		
		nose_min_x = int(landmarks[31][0] - nose_right_dist)   
		nose_max_x = int(landmarks[35][0] + nose_left_dist) 
		nose_min_y = cut_h 
		nose_max_y = int(landmarks[nose_index, 1]) 

		# Clear the nose area from the mask using a rectangle
		mask_nose = np.ones((height, width, 1), dtype=dtype)
		cv2.rectangle(mask_nose, (nose_min_x, nose_min_y), (nose_max_x, nose_max_y), color=(0), thickness=cv2.FILLED)

		mask_nose = 1 - mask_nose
		mask_tmp = mask_tmp + mask_nose
		mask_tmp = 1 - mask_tmp
		masks.append(mask_tmp)

	masks = np.asarray(masks)
	
	return torch.from_numpy(masks).squeeze(-1)


def create_masks_from_landmarks_full_size(
	landmarks_batch, image_height, image_width, start_index=48, end_index=68, offset=0
):
	"""
	Efficiently creates a batch of masks using vectorized operations where each mask has ones from the highest
	landmark in the specified range (adjusted by an offset) to the bottom of the image, and zeros otherwise.

	Parameters:
	- landmarks_batch (np.array): An array of shape (B, 68, 2) containing facial landmarks for multiple samples.
	- image_height (int): The height of the image for which masks are created.
	- image_width (int): The width of the image for which masks are created.
	- start_index (int): The starting index of the range to check (inclusive).
	- end_index (int): The ending index of the range to check (inclusive).
	- offset (int): An offset to add or subtract from the y-coordinate of the highest landmark.

	Returns:
	- np.array: An array of masks of shape (B, image_height, image_width) for each batch.
	"""
	# Extract the y-coordinates for the specified range across all batches
	y_coords = landmarks_batch[:, start_index : end_index + 1, 1]

	# Find the index of the minimum y-coordinate in the specified range for each batch
	min_y_indices = np.argmin(y_coords, axis=1)

	# Gather the highest landmarks' y-coordinates using the indices found
	highest_y_coords = y_coords[np.arange(len(y_coords)), min_y_indices]

	if abs(offset) < 1 and abs(offset) > 0:
		offset = int(offset * image_height)

	# Apply the offset to the highest y-coordinate
	adjusted_y_coords = highest_y_coords + offset

	# Clip the coordinates to stay within image boundaries
	adjusted_y_coords = np.clip(adjusted_y_coords, 0, image_height - 1)

	# Use broadcasting to create a mask without loops
	# Create a range of indices from 0 to image_height - 1
	all_indices = np.arange(image_height)

	# Compare each index in 'all_indices' to each 'adjusted_y_coord' in the batch
	# 'all_indices' has shape (image_height,), we reshape to (1, image_height) to broadcast against (B, 1)
	mask_2d = (all_indices >= adjusted_y_coords[:, None]).astype(int)

	# Extend the 2D mask to a full 3D mask of size (B, image_height, image_width)
	full_mask = np.tile(mask_2d[:, :, np.newaxis], (1, 1, image_width))

	return torch.from_numpy(full_mask)


def expand_polygon(polygon, expand_size):
	"""
	Expands the polygon outward by a specified number of pixels.

	Parameters:
	- polygon (list of tuples): The polygon points as (x, y).
	- expand_size (int): The number of pixels to expand the polygon outward.

	Returns:
	- expanded_polygon (list of tuples): The expanded polygon points as (x, y).
	"""
	if expand_size == 0:
		return polygon

	# Calculate centroid of the polygon
	centroid_x = sum([point[0] for point in polygon]) / len(polygon)
	centroid_y = sum([point[1] for point in polygon]) / len(polygon)

	# Expand each point outward from the centroid
	expanded_polygon = []
	for x, y in polygon:
		vector_x = x - centroid_x
		vector_y = y - centroid_y
		length = np.sqrt(vector_x**2 + vector_y**2)
		if length == 0:
			expanded_polygon.append((x, y))
		else:
			new_x = x + expand_size * (vector_x / length)
			new_y = y + expand_size * (vector_y / length)
			expanded_polygon.append((int(new_x), int(new_y)))

	return expanded_polygon


def create_face_mask_from_landmarks(landmarks_batch, image_height, image_width, mask_expand=0):
	"""
	Creates a batch of masks where each mask covers the face region using landmarks.

	Parameters:
	- landmarks_batch (np.array): An array of shape (B, 68, 2) containing facial landmarks for multiple samples.
	- image_height (int): The height of the image for which masks are created.
	- image_width (int): The width of the image for which masks are created.
	- mask_expand (int): The number of pixels to expand the mask outward.

	Returns:
	- np.array: An array of masks of shape (B, image_height, image_width) for each batch.
	"""
	# Initialize an array to hold all masks
	masks = np.zeros((landmarks_batch.shape[0], image_height, image_width), dtype=np.uint8)

	if abs(mask_expand) < 1 and abs(mask_expand) > 0:
		mask_expand = int(mask_expand * image_height)

	for i, landmarks in enumerate(landmarks_batch):
		# Create a blank image for each mask
		mask = Image.new("L", (image_width, image_height), 0)
		draw = ImageDraw.Draw(mask)

		# Extract relevant landmarks for the face
		jawline_landmarks = landmarks[2:15]  # Jawline
		# upper_face_landmarks = landmarks[17:27]  # Eyebrows and top of nose bridge

		# Combine landmarks to form a polygon around the face
		# face_polygon = np.concatenate((jawline_landmarks, upper_face_landmarks[::-1]), axis=0)
		face_polygon = jawline_landmarks

		# Convert landmarks to a list of tuples
		face_polygon = [(int(x), int(y)) for x, y in face_polygon]

		# Expand the polygon if necessary
		expanded_polygon = expand_polygon(face_polygon, mask_expand)

		# Draw the polygon and fill it
		draw.polygon(expanded_polygon, outline=1, fill=1)

		# Convert mask to numpy array and add it to the batch of masks
		masks[i] = np.array(mask)

	return torch.from_numpy(masks)


ALL_FIXED_POINTS = (
    [i for i in range(0, 4)] + [i for i in range(13, 17)] + [i for i in range(27, 36)] + [36, 39, 42, 45]
)


def gaussian_kernel(sigma, width, height):
    """Create a 2D Gaussian kernel."""
    x = torch.arange(0, width, 1) - width // 2
    y = torch.arange(0, height, 1) - height // 2
    x = x.float()
    y = y.float()
    x2 = x**2
    y2 = y[:, None] ** 2
    g = torch.exp(-(x2 + y2) / (2 * sigma**2))
    return g / g.sum()


def generate_hm(landmarks, height, width, n_points="all", sigma=3):
    if n_points == "all":
        Nlandmarks = range(len(landmarks))
    elif n_points == "fixed":
        Nlandmarks = ALL_FIXED_POINTS
    elif n_points == "stable":
        Nlandmarks = [33, 36, 39, 42, 45]

    kernel = gaussian_kernel(sigma, width, height)
    hm = torch.zeros((height, width))
    for I in Nlandmarks:
        x0, y0 = landmarks[I]
        x0, y0 = int(x0), int(y0)
        left, right = max(0, x0 - width // 2), min(width, x0 + width // 2)
        top, bottom = max(0, y0 - height // 2), min(height, y0 + height // 2)
        hm[top:bottom, left:right] += kernel[
            max(0, -y0 + height // 2) : min(height, height - y0 + height // 2),
            max(0, -x0 + width // 2) : min(width, width - x0 + width // 2),
        ]
    # Normalize the heatmap to have values between 0 and 1
    max_val = hm.max()
    if max_val > 0:
        hm /= max_val
    return hm


def get_heatmap(landmarks, image_size, or_im_size, n_points="stable", sigma=4):
    stack = []
    seq_length = landmarks.shape[0]
    if or_im_size[0] != image_size[0] or or_im_size[1] != image_size[1]:
        landmarks = scale_landmarks(landmarks, or_im_size, image_size)
    gen_single_heatmap = partial(
        generate_hm,
        height=image_size[0],
        width=image_size[1],
        n_points=n_points,
        sigma=sigma,
    )
    for i in range(seq_length):
        stack.append(gen_single_heatmap(landmarks[i]))

    return torch.stack(stack, axis=0).unsqueeze(0)  # (1, seq_length, height, width)


def scale_landmarks(landmarks, original_size, target_size):
	"""
	Scale landmarks from original size to target size.

	Parameters:
	- landmarks (np.array): An array of shape (N, 2) containing facial landmarks.
	- original_size (tuple): The size (height, width) for which the landmarks are currently scaled.
	- target_size (tuple): The size (height, width) to which landmarks should be scaled.

	Returns:
	- scaled_landmarks (np.array): Scaled landmarks.
	"""
	scale_y = target_size[0] / original_size[0]
	scale_x = target_size[1] / original_size[1]
	scaled_landmarks = landmarks * np.array([scale_x, scale_y])
	return scaled_landmarks.astype(int)


def draw_kps_image(
    image_shape, original_size, landmarks, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], rgb=True, pts_width=4
):
    stick_width = pts_width
    limb_seq = np.array([[0, 2], [1, 2]])
    kps = landmarks[[36, 45, 33], :]
    kps = scale_landmarks(kps, original_size, image_shape)
    if not rgb:  # Grayscale image
        canvas = np.zeros((image_shape[0], image_shape[1], 1))
        color_mode = "grayscale"
    else:  # Color image
        canvas = np.zeros((image_shape[0], image_shape[1], 3))
        color_mode = "color"

    polygon_cache = {}

    for index in limb_seq:
        color = color_list[index[0]]
        if color_mode == "grayscale":
            color = (int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]),)  # Convert to grayscale intensity

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))

        cache_key = (color, int(np.mean(x)), int(np.mean(y)), int(length / 2), int(angle))
        if cache_key not in polygon_cache:
            polygon_cache[cache_key] = cv2.ellipse2Poly(
                (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stick_width), int(angle), 0, 360, 1
            )

        polygon = polygon_cache[cache_key]
        cv2.fillConvexPoly(canvas, polygon, [int(c * 0.6) for c in color])

    for idx, kp in enumerate(kps):
        if color_mode == "grayscale":
            color = (int(0.299 * color_list[idx][2] + 0.587 * color_list[idx][1] + 0.114 * color_list[idx][0]),)
        else:
            color = color_list[idx]
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), pts_width, color, -1)

    return canvas.transpose(2, 0, 1)


def create_landmarks_image(
    landmarks, original_size=(772, 772), target_size=(772, 772), point_size=3, n_points="all", dim=3
):
    """
    Creates an image of landmarks on a black background using efficient NumPy operations.

	Parameters:
	- landmarks (np.array): An array of shape (68, 2) containing facial landmarks.
	- image_size (tuple): The size of the output image (height, width).
	- point_size (int): The radius of each landmark point in pixels.

    Returns:
    - img (np.array): An image array with landmarks plotted.
    """
    if n_points == "all":
        indexes = range(len(landmarks))
    elif n_points == "fixed":
        indexes = ALL_FIXED_POINTS
    elif n_points == "stable":
        indexes = [33, 36, 39, 42, 45]

    landmarks = landmarks[indexes]

    img = np.zeros(target_size, dtype=np.uint8)
    landmarks = scale_landmarks(landmarks, original_size, target_size)

	# Ensure the landmarks are in bounds and integer
    landmarks = np.clip(landmarks, [0, 0], [target_size[1] - 1, target_size[0] - 1]).astype(int)

	# Get x and y coordinates from landmarks
    x, y = landmarks[:, 0], landmarks[:, 1]

	# Define a grid offset based on point_size around each landmark
    offset = np.arange(-point_size // 2, point_size // 2 + 1)
    grid_x, grid_y = np.meshgrid(offset, offset, indexing="ij")

	# Calculate the full set of x and y coordinates for the points
    full_x = x[:, np.newaxis, np.newaxis] + grid_x[np.newaxis, :, :]
    full_y = y[:, np.newaxis, np.newaxis] + grid_y[np.newaxis, :, :]

	# Clip the coordinates to stay within image boundaries
    full_x = np.clip(full_x, 0, target_size[1] - 1)
    full_y = np.clip(full_y, 0, target_size[0] - 1)

	# Flatten the arrays to use them as indices
    full_x = full_x.ravel()
    full_y = full_y.ravel()

	# Set the points in the image
    img[full_y, full_x] = 255   
    return np.stack([img] * dim, axis=0)


def trim_pad_audio(audio, sr, max_len_sec=None, max_len_raw=None):
	len_file = audio.shape[-1]

	if max_len_sec or max_len_raw:
		max_len = max_len_raw if max_len_raw is not None else int(max_len_sec * sr)
		if len_file < int(max_len):
			# dummy = np.zeros((1, int(max_len_sec * sr) - len_file))
			# extened_wav = np.concatenate((audio_data, dummy[0]))
			extened_wav = torch.nn.functional.pad(audio, (0, int(max_len) - len_file), "constant")
		else:
			extened_wav = audio[:, : int(max_len)]
	else:
		extened_wav = audio

	return extened_wav


def ssim_to_bin(ssim_score):
	# Normalize the SSIM score to a 0-100 scale
	normalized_diff_ssim = (1 - ((ssim_score + 1) / 2)) * 100
	# Assign to one of the 100 bins
	bin_index = float(min(np.floor(normalized_diff_ssim), 99))
	return bin_index
