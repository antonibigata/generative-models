# pyre-strict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
# from bc.face_api.analysis import ImageAnalysis
# from bc.face_api.api import FaceAPILocal


def im2tensor(im: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(im).permute(2, 0, 1)


def tensor2im(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).numpy()

"""
steepness: how quickly the sigmoid function transitions from 0 to 1, A higher steepness value results to faster transition
displacement: This parameter controls the horizontal shift of the sigmoid curve along the x-axis. Positive displacement values shift the curve to the right, while negative displacement values shift it to the left.
"""
def sigmoid_blending(
    x: np.ndarray, steepness: int = -20, displacement: int = 150
) -> Union[float, np.ndarray]:
    return 1 / (1 + displacement * np.exp(steepness * x))

def adjust_bbox_size(blending_box, a, b):
    bbox_height, bbox_width, _ = blending_box.shape
    a_height, a_width, _ = a.shape
    b_height, b_width, _ = b.shape
    
    # Adjust height if necessary
    if bbox_height != min(a_height, b_height):
        bbox_height = min(a_height, b_height)
    
    # Adjust width if necessary
    if bbox_width != min(a_width, b_width):
        bbox_width = min(a_width, b_width)
    
    # Adjust blending_box size
    new_bbox = blending_box[:bbox_height, :bbox_width, :]
    
    return new_bbox


def blending_matrix(
    w: int, h: int, sigmoid_steepness: int, sigmoid_displacement: int
) -> np.ndarray:

    el = (
        list(
            # pyre-fixme: Incompatible parameter type [6]: In call `list.__init__`, for 1st positional argument, expected `Iterable[Variable[_T]]` but got `Union[float, ndarray]`
            sigmoid_blending(
                np.linspace(0, 1, w // 2),
                steepness=sigmoid_steepness,
                displacement=sigmoid_displacement,
            ),
        )
        # 1st positional argument, expected `Iterable[Variable[_T]]` but got `Union[float, ndarray]`.
        + list(
            reversed(
                # pyre-fixme: Incompatible parameter type [6]: In call `list.__init__`, for
                sigmoid_blending(
                    np.linspace(0, 1, w // 2 + 1),
                    steepness=sigmoid_steepness,
                    displacement=sigmoid_displacement,
                )
            )
        )[1:]
    )
    w = len(el)  # to make sure we have the correct length
    
    # pyre-ignore
    rows = np.vstack([el for _ in range(h)])
    el = (
        # pyre-fixme: Incompatible parameter type [6]: In call `list.__init__`, for
        # 1st positional argument, expected `Iterable[Variable[_T]]` but got `Union[float, ndarray]`.
        list(sigmoid_blending(np.linspace(0, 1, h // 2 + 1)))
        # pyre-fixme: Incompatible parameter type [6]: In call `list.__init__`, for
        # 1st positional argument, expected `Iterable[Variable[_T]]` but got `Union[float, ndarray]`.
        + list(reversed(sigmoid_blending(np.linspace(0, 1, h // 2))))[1:]
    )
    # pyre-ignore
    cols = np.vstack([el for _ in range(w)]).transpose()
    matrix = rows * cols
    matrix = matrix / matrix.max()
    return matrix


