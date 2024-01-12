import warnings
from typing import Union, List, Dict
from omegaconf import DictConfig, ListConfig
import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode
from einops import repeat, rearrange
from .base import AbstractMapper
from ..datapipeline import instantiate, time_measure, timeout_wrapper
import io


class Rescaler(AbstractMapper):
    def __init__(
        self,
        key: Union[List[str], ListConfig, str] = "jpg",
        isfloat: bool = True,
        strict: bool = True,
        *args,
        **kwargs,
    ):
        """

        :param key: the key indicating the sample
        :param isfloat: bool indicating whether input is float in [0,1]
        or uint in [0.255]
        """
        # keeping name of first argument to be 'key' for the sake of backwards compatibility
        super().__init__(*args, **kwargs)
        if isinstance(key, str):
            key = [key]
        self.keys = set(key)
        self.isfloat = isfloat
        self.strict = strict
        self.has_warned = [False, False]

    @timeout_wrapper
    @time_measure("Rescaler")
    def __call__(self, sample: Dict) -> Dict:
        """

        :param sample: Dict containing the speficied key, which should be a torch.Tensor or numpy array
        :return:
        """
        if self.skip_this_sample(sample):
            return sample
        if not any(map(lambda x: x in sample, self.keys)):
            if self.strict:
                raise KeyError(f"None of {self.keys} in current sample with keys {list(sample.keys())}")
            else:
                if not self.has_warned[0]:
                    self.has_warned[0] = True
                    warnings.warn(
                        f"None of {self.keys} contained in sample"
                        f"(for sample with keys {list(sample.keys())}). "
                        f"Sample is returned unprocessed since strict mode not enabled"
                    )
                return sample

        matching_keys = set(self.keys.intersection(sample))
        if len(matching_keys) > 1:
            if self.strict:
                raise ValueError(
                    f"more than one matching key of {self.keys} in sample {list(sample.keys())}. This should not be the case"
                )
            else:
                if not self.has_warned[1]:
                    warnings.warn(
                        f"more than one matching key of {self.keys} in sample {list(sample.keys())}."
                        f" But strict mode disabled, so returning sample unchanged"
                    )
                    self.has_warned[1] = True
                return sample

        key = matching_keys.pop()

        if self.isfloat:
            sample[key] = sample[key] * 2 - 1.0
        else:
            sample[key] = sample[key] / 127.5 - 1.0

        return sample


class SelectTuple(AbstractMapper):
    def __init__(self, key: str, index: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.index = index

    @timeout_wrapper
    @time_measure("SelectTuple")
    def __call__(self, sample: Dict) -> Dict:
        if self.skip_this_sample(sample):
            return sample
        sample[self.key] = sample[self.key][self.index]
        return sample


class ToSVDFormat(AbstractMapper):
    def __init__(
        self,
        key: str = "mp4",
        n_frames=14,
        resize_size=256,
        cond_noise=[-3.0, 0.5],
        motion_id=255.0,
        audio_key=None,
        fps=25,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.key = key
        # self.n_frames = n_frames + 1  # +1 for the conditional frame
        self.n_frames = n_frames
        self.resize_size = resize_size
        self.cond_noise = cond_noise
        self.motion_id = motion_id
        self.audio_key = audio_key
        self.fps = fps

        self.resizer = TT.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC, antialias=True)

    @timeout_wrapper
    @time_measure("ToSVDFormat")
    def __call__(self, sample: Dict) -> Dict:
        if self.skip_this_sample(sample):
            return sample
        video = sample[self.key]
        del sample[self.key]
        video = video.permute(0, 3, 1, 2)

        # Select n consecutive frames
        start = np.random.randint(0, video.shape[0] - self.n_frames)
        video = video[start : start + self.n_frames]

        # Resize
        video = self.resizer(video)

        # Normalize
        video = (video / 255.0) * 2 - 1.0
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        cond = video[:, 0]
        # cond = repeat(cond, "c h w -> c t h w", t=self.n_frames)
        # video = video[:, 1:]

        # Add noise to conditional frame
        if isinstance(self.cond_noise, ListConfig):
            cond_noise = (self.cond_noise[0] + self.cond_noise[1] * torch.randn((1,))).exp()
            noisy_cond = cond + cond_noise * torch.randn_like(cond)
        else:
            noisy_cond = cond + self.cond_noise * torch.randn_like(cond)
            cond_noise = self.cond_noise

        # Add audio
        if self.audio_key is not None:
            audio = sample[self.audio_key]
            audio = audio[start : start + self.n_frames]
            # sample["audio_emb"] = audio[1:]
            sample["audio_emb"] = audio

        sample["frames"] = video
        sample["cond_frames"] = noisy_cond
        sample["cond_frames_without_noise"] = cond
        sample["cond_aug"] = cond_noise
        sample["motion_bucket_id"] = torch.tensor([self.motion_id])
        sample["fps_id"] = torch.tensor([self.fps])
        sample["num_video_frames"] = self.n_frames - 1
        sample["image_only_indicator"] = torch.zeros(self.n_frames - 1)
        return sample


class TorchVisionImageTransforms(AbstractMapper):
    def __init__(
        self,
        transforms: Union[Union[Dict, DictConfig], ListConfig],
        key: str = "jpg",
        strict: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strict = strict
        self.key = key
        chained_transforms = []

        if isinstance(transforms, (DictConfig, Dict)):
            transforms = [transforms]

        for trf in transforms:
            trf = instantiate(trf)
            chained_transforms.append(trf)

        self.transform = TT.Compose(chained_transforms)

    @timeout_wrapper
    @time_measure("TorchVisionImageTransforms")
    def __call__(self, sample: Dict) -> Union[Dict, None]:
        if self.skip_this_sample(sample):
            return sample
        if self.key not in sample:
            if self.strict:
                del sample
                return None
            else:
                return sample
        sample[self.key] = self.transform(sample[self.key])
        return sample


class AddOriginalImageSizeAsTupleAndCropToSquare(AbstractMapper):
    """
    Adds the original image size as params and crops to a square.
    Also adds cropping parameters. Requires that no RandomCrop/CenterCrop has been called before
    """

    def __init__(
        self,
        h_key: str = "original_height",
        w_key: str = "original_width",
        image_key: str = "jpg",
        use_data_key: bool = True,
        data_key: str = "json",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.h_key, self.w_key = h_key, w_key
        self.image_key = image_key
        self.data_key = data_key
        self.use_data_key = use_data_key

    @timeout_wrapper
    @time_measure("AddOriginalImageSizeAsTupleAndCropToSquare")
    def __call__(self, x: Dict) -> Dict:
        if self.skip_this_sample(x):
            return x
        if self.use_data_key:
            h, w = map(lambda y: x["json"][y], (self.h_key, self.w_key))
        else:
            h, w = map(lambda y: x[y], (self.h_key, self.w_key))
        x["original_size_as_tuple"] = torch.tensor([h, w])
        jpg = x[self.image_key]
        if not isinstance(jpg, torch.Tensor) and jpg.shape[0] not in [1, 3]:
            raise ValueError(
                f"{self.__class__.__name__} requires input image to be a torch.Tensor with channels-first"
            )
        # x['jpg'] should be chw tensor  in [-1, 1] at this point
        size = min(jpg.shape[1], jpg.shape[2])
        delta_h = jpg.shape[1] - size
        delta_w = jpg.shape[2] - size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
        x[self.image_key] = TT.functional.crop(jpg, top=top, left=left, height=size, width=size)
        x["crop_coords_top_left"] = torch.tensor([top, left])
        return x
