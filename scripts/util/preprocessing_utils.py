# pyre-strict
from typing import List, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F


def find_center_of_bbox(bbox: torch.Tensor) -> np.ndarray:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return np.array([x, y])

def clamp_value(x: float, min_x: float, max_x: float) -> float:
    return max(min_x, min(x, max_x))

def resize_frame(img_tensor:torch.Tensor, size:int=1024, mode='bicubic') -> torch.Tensor:
    return F.interpolate(img_tensor.permute(2, 0, 1).unsqueeze(0), size=(size, size), mode=mode, align_corners=False).squeeze(0).permute(1, 2, 0)

def crop_and_save_video(
    frames: torch.Tensor,
    bboxes: torch.Tensor,
    keypoints: torch.Tensor,
    crop_center: Tuple[int, int],
    crop_size: int,
    output_size: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    cropped_frames = []
    cropped_bboxes = []
    cropped_keypoints = []

    for frame_idx, frame in enumerate(frames):
        x_center, y_center = crop_center
        start_x = max(x_center - crop_size // 2, 0)
        start_y = max(y_center - crop_size // 2, 0)
        
#         import pdb
#         pdb.set_trace()     
        
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]

        if cropped_frame.shape[0] != crop_size or cropped_frame.shape[1] != crop_size:
            # If the crop goes out of bounds, fill the rest with black pixels
            new_frame = np.zeros((crop_size, crop_size, 3), np.uint8)
            new_frame[: cropped_frame.shape[0], : cropped_frame.shape[1]] = (
                cropped_frame
            )
            cropped_frame = torch.from_numpy(new_frame)
        
        
        cropped_frames.append(cropped_frame)
        # Adjust bounding boxes
        bbox = bboxes[frame_idx]
        frame_cropped_bboxes = []

        xmin, ymin, xmax, ymax = bbox
        cropped_xmin = max(xmin - start_x, 0)
        cropped_ymin = max(ymin - start_y, 0)
        cropped_xmax = max(min(xmax - start_x, crop_size), cropped_xmin)
        cropped_ymax = max(min(ymax - start_y, crop_size), cropped_ymin)

        frame_cropped_bboxes.append(
            [cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax]
        )

        cropped_bboxes.append(frame_cropped_bboxes)

        # Adjust keypoints
        frame_keypoints = keypoints[frame_idx]
        frame_cropped_keypoints = []
        for point in frame_keypoints:
            px, py = point
            cropped_px = px - start_x
            cropped_py = py - start_y
            cropped_px_clamped = clamp_value(cropped_px, 0, crop_size)
            cropped_py_clamped = clamp_value(cropped_py, 0, crop_size)

            frame_cropped_keypoints.append((cropped_px, cropped_py))

        cropped_keypoints.append(frame_cropped_keypoints)

    return cropped_frames, cropped_bboxes, cropped_keypoints


def adjust_bbox(
    bbox: torch.Tensor, scale_factor: float, max_width: int, max_height: int
) -> Tuple[int, int, int, int]:
    """Adjust the bounding box considering the scale and ensuring it's within image boundaries."""
    x, y, xmax, ymax = bbox
    w = xmax - x
    h = ymax - y
    """Adjust the bounding box considering the scale and ensuring it's within image boundaries."""
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    new_x = max(x + w // 2 - new_w // 2, 0)
    new_y = max(y + h // 2 - new_h // 2, 0)
    # Adjust if the bbox goes beyond the frame boundaries
    new_x = min(new_x, max_width - new_w)
    new_y = min(new_y, max_height - new_h)
    return int(new_x), int(new_y), int(new_w), int(new_h)


def stabilize_bbox_eg(
    bboxes: List[float],
    window_size: int = 7,
    max_sigma: float = 8,
    min_sigma: float = 2,
    threshold: float = 10,
) -> List[Tuple[int, int, int, int]]:
    """Apply adaptive Gaussian smoothing to the bounding box dimensions based on historical movement differences."""
    # if len(bboxes) < window_size:
    #     return bboxes

    x, y, w, h = zip(*bboxes)
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)
    h = np.array(h)

    # Function to calculate sigma based on the differences
    def calculate_sigma(values: np.ndarray) -> float:
        # Calculate absolute differences from each point to its three previous points
        diffs = np.abs(
            [
                values[i] - values[i - j]
                for i in range(3, len(values))
                for j in range(1, 4)
            ]
        )
        mean_diff = np.mean(diffs) if len(diffs) > 0 else 0

        # Define sigma based on mean_diff
        if mean_diff == 0:
            return max_sigma
        elif mean_diff >= threshold:
            return min_sigma
        else:
            # Non-linear transformation for sigma
            # If mean_diff is half of threshold, sigma should be 0.75*(max_sigma - min_sigma) + min_sigma
            # Use a square function to adjust the rate of change
            normalized_diff = mean_diff / threshold
            factor = 1 - normalized_diff**2  # Square to make changes less linear
            sigma = factor * (max_sigma - min_sigma) + min_sigma
            return sigma

    sigma_x = calculate_sigma(x)
    sigma_y = calculate_sigma(y)

    # Apply Gaussian filter with adaptive sigma
    x_smooth = gaussian_filter1d(x, sigma=sigma_x)
    y_smooth = gaussian_filter1d(y, sigma=sigma_y)
    w_smooth = gaussian_filter1d(
        w, sigma=min_sigma
    )  # Assume less variability in width and height
    h_smooth = gaussian_filter1d(h, sigma=min_sigma)

    # Convert smoothed values back to integers for pixel coordinates
    return list(
        zip(
            map(int, x_smooth),
            map(int, y_smooth),
            map(int, w_smooth),
            map(int, h_smooth),
        )
    )


def overlay_bboxes_torch(
    frames: torch.Tensor, bboxes: torch.Tensor, output_path: str, fps: int = 30
) -> None:

    # Setup the video writer
    height, width = frames.shape[2], frames.shape[3]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height)
    )

    try:
        # Convert frames to NumPy arrays in BGR format for OpenCV
        for frame, bbox in zip(frames, bboxes):
            frame_np = frame.permute(1, 2, 0).numpy()  # Convert to HxWxC
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

            if bbox.nelement() == 0:
                # Skip drawing if no bbox is provided for a frame
                continue

            # Extract bounding box coordinates
            if bbox.shape == (1, 4):
                x, y, w, h = bbox[0]
                cv2.rectangle(
                    frame_bgr, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2
                )
            else:
                raise ValueError(
                    "Bounding box format is incorrect; expected shape (1, 4)"
                )

            # Write the modified frame to video
            out.write(frame_bgr)
    finally:
        out.release()
        # cv2.destroyAllWindows()
        print("Video processing complete, output saved to:", output_path)


def overlay_landmarks_torch(
    frames: torch.Tensor, landmarks: torch.Tensor, output_path: str, fps: int = 60
) -> None:
    # Ensure the frames tensor is in the correct format (numpy array, HxWxC)
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(
            0, 2, 3, 1
        ).numpy()  # Convert from CxHxW to HxWxC if needed

    # Ensure landmarks are numpy array
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.numpy()

    # Define the codec and create VideoWriter object
    frame_height, frame_width, _ = frames.shape[1:]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (frame_width, frame_height)
    )

    for frame_index, frame in enumerate(frames):
        # Convert frame to correct color format for cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw landmarks on the frame if they exist for the current frame
        if frame_index < len(landmarks) and landmarks[frame_index] is not None:
            for x, y in landmarks[frame_index]:
                cv2.circle(
                    frame, (int(x), int(y)), 2, (0, 255, 0), -1
                )  # Draw green dot

        out.write(frame)

    # Release everything when job is finished
    out.release()
    # cv2.destroyAllWindows()
    print("Video processing complete, output saved to:", output_path)


def overlay_landmarks(
    video: np.ndarray, landmarks: np.ndarray, output_path: str, fps: int = 30
) -> None:

    # Assume video dimensions and number of frames from the video array shape
    num_frames, frame_height, frame_width, channels = video.shape

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (frame_width, frame_height)
    )

    # Iterate over each frame and the corresponding landmarks
    for frame_index, frame in enumerate(video):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_index < len(landmarks):
            # Draw landmarks on the frame
            for x, y in landmarks[frame_index]:

                cv2.circle(
                    frame, (int(x), int(y)), 2, (0, 255, 0), -1
                )  # Draw green dot

        # Write the modified frame to the video
        out.write(frame)

    # Release everything when job is finished
    out.release()
    # cv2.destroyAllWindows()
    print("Video processing complete, output saved to:", output_path)

def generate_video(frames: np.ndarray, output_path: str):

    if isinstance(frames, torch.Tensor):
        frames = frames.permute(
            0, 2, 3, 1
        ).numpy()  # Conv

    frame_height, frame_width, _ = frames.shape[1:]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), 25, (frame_width, frame_height)
    )
    try:
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = frame.astype('uint8')
            out.write(frame)
    finally:
        out.release()
        # cv2.destroyAllWindows()
        print("Video processing complete, output saved to:", output_path)

def overlay_bboxes(
    frames: np.ndarray, bboxes: np.ndarray, output_path: str, fps: int = 30
) -> None:

    height, width, _ = frames.shape[1], frames.shape[2], frames.shape[3]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height)
    )

    try:
        for frame, bbox in zip(frames, bboxes):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if bbox.size == 0:
                # Skip drawing if no bbox is provided for a frame
                continue

            frame_bgr = frame
            if bbox.shape == (4,):
                x, y, w, h = bbox
                cv2.rectangle(
                    frame_bgr, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2
                )
            else:
                raise ValueError(
                    f"Bounding box format is incorrect; expected shape (4,), but {bbox.shape}"
                )

            out.write(frame_bgr)
    finally:
        out.release()
        # cv2.destroyAllWindows()
        print("Video processing complete, output saved to:", output_path)
