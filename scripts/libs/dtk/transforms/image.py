import numbers
import random
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F


def _is_tensor_image(image):
    if not torch.is_tensor(image):
        raise TypeError("image should be Tesnor. Got %s" % type(image))

    if not image.ndimension() == 3:
        raise ValueError("image should be 3D. Got %dD" % image.dim()) # C x H x W

    return True


def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode)



def to_tensor(image):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    Args:
        image (torch.tensor, dtype=torch.uint8): Size is (H, W, C)
    Return:
        image (torch.tensor, dtype=torch.float): Size is (C, H, W)
    """
    # _is_tensor_image(image)
    if not torch.is_tensor(image):
        image = torch.from_numpy(image)
    if not image.dtype == torch.uint8:
        raise TypeError("image tensor should have data type uint8. Got %s" % str(image.dtype))
    return image.float().permute(2, 0, 1) / 255.0


def normalize(image, mean, std, inplace=False):
    """
    Args:
        image (torch.tensor): Video clip to be normalized. Size is (C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized image (torch.tensor): Size is ( C, H, W)
    """
    # assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
        image = image.clone()
    mean = torch.as_tensor(mean).type_as(image)
    std = torch.as_tensor(std).type_as(image)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image



class NormalizeImage(object):
    """
    Normalize the image by mean subtraction
    and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image):
        """
        Args:
            climageip (torch.tensor): image to be normalized. Size is (C, H, W)
        """
        return normalize(image, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, inplace={2})".format(self.mean, self.std, self.inplace)


class ToTensorImage(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of image tensor
    """

    def __init__(self):
        pass

    def __call__(self, image):
        """
        Args:
            image (torch.tensor, dtype=torch.uint8): Size is (H, W, C)
        Return:
            image (torch.tensor, dtype=torch.float): Size is (C, H, W)
        """
        return to_tensor(image)

    def __repr__(self):
        return self.__class__.__name__


