# pyre-strict
import json
from typing import Any, Dict, Optional, Tuple

import cv2

import decord

import numpy as np
import torch
import torch.nn as nn

# from bc.face_api.batch import FaceAPIBatch
# from bc.face_api.config import get_config
# from bc.utils.data.path_manager import BCPathManager
# from bc.utils.filtering import RTSSmoother
from torch.nn import functional as F
from torchvision import transforms


def solve_affine(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    transforms = []
    for src_points, dst_points in zip(src.cpu(), dst.cpu()):
        src_np = src_points.numpy().astype(np.float32)
        dst_np = dst_points.numpy().astype(np.float32)
        transform = cv2.estimateAffinePartial2D(
            src_np,
            dst_np,
            method=cv2.LMEDS,
        )[0]
        if transform is None:
            transforms.append(np.zeros((2, 3)))
        else:
            transforms.append(transform)

    return torch.from_numpy(np.stack(transforms).astype(np.float32)).to(src.device)


def normalize_coordinates(coords: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    W, H = img_shape
    coords = coords.clone().float()
    coords[..., 0] = 2.0 * (coords[..., 0] / W) - 1.0
    coords[..., 1] = 2.0 * (coords[..., 1] / H) - 1.0
    return coords


class EmotionDetector(nn.Module):
    def __init__(
        self,
        model_path: str,
        chunk_size: int = 32,
        device: Optional[torch.device] = None,
        provide_landmarks: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        # self.path_mgr = BCPathManager()
        self.chunk_size = chunk_size
        self._load_model_(model_path)
        self.model.eval()
        self.stable_pt_ids = (28, 33, 36, 39, 42, 45, 48, 54)

        if not provide_landmarks:
            # self.face_api: Optional[FaceAPIBatch] = FaceAPIBatch(get_config("batch_2d_fan"), device=device)
            pass
        else:
            self.face_api = None

        self.transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        decord.bridge.set_bridge("torch")
        self.to(device)

    def _load_model_(self, model_file_path: str) -> None:
        if model_file_path.startswith("manifold://"):
            model_file_path = self.path_mgr.get_local_path(model_file_path)

        extra_files = {
            "info": "",
            "mean_face": "",
        }
        self.model = torch.jit.load(
            model_file_path,
            map_location=torch.device("cpu"),
            _extra_files=extra_files,
        )
        self.info: Dict[str, Any] = json.loads(extra_files["info"])

        self.register_buffer(
            "mean_face",
            torch.from_numpy(np.reshape(np.frombuffer(extra_files["mean_face"], dtype=np.float32), (100, 2))),
        )

    def process_video(self, video_path: str, landmarks: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        vr = decord.VideoReader(video_path)
        assert len(vr) > 0

        if landmarks is not None:
            assert landmarks.shape[0] == len(
                vr
            ), f"Landmarks shape {landmarks.shape[0]} does not match video length {len(vr)}"

        fps = vr.get_avg_fps()

        partial_results = []
        for i in range(0, len(vr), self.chunk_size):
            video = vr.get_batch(range(i, min(len(vr), i + self.chunk_size))).to(self.mean_face.device)
            landmarks_chunk = landmarks[i : i + self.chunk_size] if landmarks is not None else None

            with torch.no_grad():
                video = self.preprocess(video, fps=fps, landmarks=landmarks_chunk)
                partial_result = self.model(video).detach().cpu()
                partial_results.append(partial_result)

        result = torch.cat(partial_results)
        scores = result[:, : len(self.info["classes"])]
        return {
            "labels": [self.info["classes"][str(frame_score.item())] for frame_score in scores.argmax(-1)],
            "valence": result[:, self.info["Valence"]],
            "arousal": result[:, self.info["Arousal"]],
            "scores": scores,
        }

    def preprocess(
        self,
        batch: torch.Tensor,
        fps: Optional[float] = None,
        landmarks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            aligned_batch = self.align_faces(
                batch.view(-1, batch.shape[-3], batch.shape[-2], batch.shape[-1]),
                fps=fps,
                landmarks=landmarks,
            )
            return self.transforms(aligned_batch)

    def align_faces(
        self,
        batch: torch.Tensor,
        fps: Optional[float] = None,
        landmarks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            num_stable_landmarks = len(self.stable_pt_ids)
            if landmarks is None:
                assert self.face_api is not None
                np_landmarks = self.face_api.process_batch(batch)["landmarks"]
            else:
                np_landmarks = landmarks.numpy()

            # Fill in missing landmarks with NaNs
            # for i in range(0, len(np_landmarks)):
            #     if np_landmarks[i].shape[0] > 1:
            #         np_landmarks[i] = np_landmarks[i][0].reshape(1, np_landmarks[i].shape[1], np_landmarks[i].shape[2])
            #     if np_landmarks[i].size == 0:
            #         np_landmarks[i] = np.full([1, 100, 2], np.nan)

            np_landmarks = np_landmarks[:, self.stable_pt_ids]
            # if fps is not None:
            #     smoother = RTSSmoother(fps=fps, detector_accuracy=4, process_noise=600)
            #     np_landmarks = smoother(np_landmarks)

            all_stable_landmarks = torch.from_numpy(np_landmarks[:, :, :2]).reshape((-1, num_stable_landmarks, 2))

            target_face = (
                self.mean_face[self.stable_pt_ids, :]
                .unsqueeze(0)
                .repeat((all_stable_landmarks.shape[0], 1, 1))
                .to(batch.device)
            )
            landmark_transforms = solve_affine(
                normalize_coordinates(target_face, (self.info["img_size"], self.info["img_size"])),
                normalize_coordinates(all_stable_landmarks, (batch.shape[2], batch.shape[1])),
            )

            frames = batch.permute(0, 3, 1, 2)
            frames = frames / 255

            grid = F.affine_grid(
                landmark_transforms.cpu(),
                [frames.shape[0], 3, self.info["img_size"], self.info["img_size"]],
                align_corners=False,
            ).to(frames.device)
            warped_batch = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros")

        return warped_batch

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            result = self.model(image)
            scores = result[:, : len(self.info["classes"])]
            return {
                "labels": [self.info["classes"][str(frame_score.item())] for frame_score in scores.argmax(-1)],
                "valence": result[:, self.info["Valence"]],
                "arousal": result[:, self.info["Arousal"]],
                "scores": scores,
            }
