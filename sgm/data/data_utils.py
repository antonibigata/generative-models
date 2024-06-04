import torch
import numpy as np


ALL_FIXED_POINTS = (
    [i for i in range(0, 4)] + [i for i in range(13, 17)] + [i for i in range(27, 36)] + [36, 39, 42, 45]
)
from functools import partial
import torch


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
