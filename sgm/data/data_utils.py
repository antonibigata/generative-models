import torch
import numpy as np


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


def create_landmarks_image(landmarks, original_size=(772, 772), target_size=(772, 772), point_size=3):
    """
    Creates an image of landmarks on a black background using efficient NumPy operations.

    Parameters:
    - landmarks (np.array): An array of shape (68, 2) containing facial landmarks.
    - image_size (tuple): The size of the output image (height, width).
    - point_size (int): The radius of each landmark point in pixels.

    Returns:
    - img (np.array): An image array with landmarks plotted.
    """
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

    return np.stack([img] * 3, axis=0)


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
