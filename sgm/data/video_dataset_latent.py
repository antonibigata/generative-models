import os
import numpy as np
from functools import partial
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import math
import decord
from einops import rearrange
from more_itertools import sliding_window
from omegaconf import ListConfig
import torchaudio
import soundfile as sf
from torchvision.transforms import RandomHorizontalFlip
from audiomentations import Compose, AddGaussianNoise, PitchShift
from safetensors.torch import load_file

torchaudio.set_audio_backend("sox_io")
decord.bridge.set_bridge("torch")


def exists(x):
    return x is not None


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


# Similar to regular video dataset but trades flexibility for speed
class VideoDataset(Dataset):
    def __init__(
        self,
        filelist,
        resize_size=None,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        video_extension=".avi",
        audio_extension=".wav",
        audio_rate=16000,
        # fps=25,
        num_frames=5,
        need_cond=True,
        step=1,
        mode="prediction",
        max_missing_audio_files=10,
        scale_audio=False,
        augment=False,
        augment_audio=False,
        use_latent=False,
        # precomputed_latent=False,
        latent_type="stable",
        latent_scale=1,  # For backwards compatibility
        from_audio_embedding=False,
        load_all_possible_indexes=False,
        audio_emb_type="wavlm",
        cond_noise=[-3.0, 0.5],
        motion_id=255.0,
        data_mean=None,
        data_std=None,
    ):
        self.audio_folder = audio_folder
        self.from_audio_embedding = from_audio_embedding
        self.audio_emb_type = audio_emb_type
        self.cond_noise = cond_noise
        precomputed_latent = latent_type
        # self.fps = fps

        assert not (exists(data_mean) ^ exists(data_std)), "Both data_mean and data_std should be provided"

        if data_mean is not None:
            data_mean = rearrange(torch.as_tensor(data_mean), "c -> c () () ()")
            data_std = rearrange(torch.as_tensor(data_std), "c -> c () () ()")
        self.data_mean = data_mean
        self.data_std = data_std
        self.motion_id = motion_id

        self.filelist = []
        self.audio_filelist = []
        missing_audio = 0
        with open(filelist, "r") as files:
            for f in files.readlines():
                f = f.rstrip()
                audio_path = f.replace(video_folder, audio_folder).replace(video_extension, audio_extension)
                if not os.path.exists(audio_path):
                    missing_audio += 1
                    print("Missing audio file: ", audio_path)
                    if missing_audio > max_missing_audio_files:
                        raise FileNotFoundError(f"Missing more than {max_missing_audio_files} audio files")
                    continue
                self.filelist += [f]
                self.audio_filelist += [audio_path]

        self.resize_size = resize_size
        if use_latent and not precomputed_latent:
            self.resize_size *= 4 if latent_type in ["stable", "ldm"] else 8
        self.scale_audio = scale_audio
        self.step = step
        self.use_latent = use_latent
        self.precomputed_latent = precomputed_latent
        self.latent_type = latent_type
        self.latent_scale = latent_scale
        self.video_ext = video_extension
        self.video_folder = video_folder

        self.augment = augment
        self.maybe_augment = RandomHorizontalFlip(p=0.5) if augment else lambda x: x
        self.maybe_augment_audio = (
            Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.002, p=0.25),
                    # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                    PitchShift(min_semitones=-1, max_semitones=1, p=0.25),
                    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.333),
                ]
            )
            if augment_audio
            else lambda x, sample_rate: x
        )
        self.maybe_augment_audio = partial(self.maybe_augment_audio, sample_rate=audio_rate)

        self.mode = mode
        if mode == "interpolation":
            need_cond = False  # Interpolation does not need condition as first and last frame becomes the condition
        self.need_cond = need_cond  # If need cond will extract one more frame than the number of frames
        # It is used for the conditional model when the condition is not on the temporal dimension
        num_frames = num_frames if not self.need_cond else num_frames + 1
        # print(f"Num frames: {num_frames}")

        # Get metadata about video and audio
        # _, self.audio_rate = torchaudio.load(self.audio_filelist[0], channels_first=False)
        vr = decord.VideoReader(self.filelist[0])
        self.video_rate = math.ceil(vr.get_avg_fps())
        print(f"Video rate: {self.video_rate}")
        self.audio_rate = audio_rate
        a2v_ratio = self.video_rate / float(self.audio_rate)
        self.samples_per_frame = math.ceil(1 / a2v_ratio)

        self.num_frames = num_frames
        self.load_all_possible_indexes = load_all_possible_indexes
        if load_all_possible_indexes:
            self._indexes = self._get_indexes(self.filelist, self.audio_filelist)
        else:
            self._indexes = list(zip(self.filelist, self.audio_filelist))
        self.total_len = len(self._indexes)

    def __len__(self):
        return self.total_len

    def _load_audio(self, filename, max_len_sec, start=None):
        audio, sr = sf.read(
            filename,
            start=math.ceil(start * self.audio_rate),
            frames=math.ceil(self.audio_rate * max_len_sec),
            always_2d=True,
        )  # e.g (16000, 1)
        audio = audio.T  # (1, 16000)
        assert sr == self.audio_rate, f"Audio rate is {sr} but should be {self.audio_rate}"
        audio = audio.mean(0, keepdims=True)
        audio = self.maybe_augment_audio(audio)
        audio = torch.from_numpy(audio).float()
        # audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.audio_rate)
        audio = trim_pad_audio(audio, self.audio_rate, max_len_sec=max_len_sec)
        return audio[0]

    def normalize_latents(self, latents):
        if self.data_mean is not None:
            # Normalize latents to 0 mean and 0.5 std
            latents = ((latents - self.data_mean) / self.data_std) * 0.5
        return latents

    def _get_frames_and_audio(self, idx):
        if self.load_all_possible_indexes:
            indexes, video_file, audio_file = self._indexes[idx]
            vr = None
        else:
            video_file, audio_file = self._indexes[idx]
            vr = decord.VideoReader(video_file)
            len_video = len(vr)
            start_idx = np.random.randint(0, len_video - self.num_frames)
            indexes = list(range(start_idx, start_idx + self.num_frames))

        audio_frames = None
        if self.use_latent and self.precomputed_latent:
            frames = load_file(video_file.replace(self.video_ext, f"_{self.latent_type}_512_latent.safetensors"))[
                "latents"
            ][indexes, :, :, :]

            if frames.shape[-1] != 64:
                print(f"Frames shape: {frames.shape}, video file: {video_file}")

            frames = rearrange(frames, "t c h w -> c t h w") * self.latent_scale
            frames = self.normalize_latents(frames)
            # if not self.from_audio_embedding:
            #     audio = self._load_audio(
            #         audio_file, max_len_sec=frames.shape[1] / self.video_rate, start=indexes[0] / self.video_rate
            #     )
            # else:
            #     audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
            #     try:
            #         audio_frames = audio[indexes, :]
            #     except IndexError as e:
            #         print(f"Index error for {audio_file}")
            #         print(f"Audio shape: {audio.shape}")
            #         print(f"Indexes: {indexes}")
            #         print(
            #             f"Frames shape: {torch.load(video_file.replace(self.video_ext, f'_{self.latent_type}_latent.pt')).shape}"
            #         )
            #         raise e
        else:
            vr = decord.VideoReader(video_file) if vr is None else vr
            frames = vr.get_batch(indexes).permute(3, 0, 1, 2).float()

        if not self.from_audio_embedding:
            audio = self._load_audio(
                audio_file, max_len_sec=frames.shape[1] / self.video_rate, start=indexes[0] / self.video_rate
            )
        else:
            audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
            audio_frames = audio[indexes, :]

        if audio_frames is None:
            audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)

        # audio_frames = audio_frames.T
        audio_frames = audio_frames[1:] if self.need_cond else audio_frames  # Remove audio of first frame

        if self.scale_audio:
            audio_frames = (audio_frames / audio_frames.max()) * 2 - 1

        if not self.use_latent or (self.use_latent and not self.precomputed_latent):
            frames = self.scale_and_crop((frames / 255.0) * 2 - 1)

        target = frames[:, 1:] if self.need_cond else frames
        if self.mode == "prediction":
            if self.use_latent:
                vr = decord.VideoReader(video_file) if vr is None else vr
                clean_cond = vr[indexes[0]].unsqueeze(0).permute(3, 0, 1, 2).float()
                clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1).squeeze(0)
                # noisy_cond = target[:, 0]
                noisy_cond = clean_cond
            else:
                clean_cond = target[:, 0]
                noisy_cond = clean_cond
        elif self.mode == "interpolation":
            if self.use_latent:
                vr = decord.VideoReader(video_file) if vr is None else vr
                clean_cond = vr.get_batch([indexes[0], indexes[-1]]).permute(3, 0, 1, 2).float()
                clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)
                # noisy_cond = torch.stack([frames[:, 0], frames[:, -1]], dim=1)
                noisy_cond = clean_cond
            else:
                clean_cond = torch.stack([frames[:, 0], frames[:, -1]], dim=1)
                noisy_cond = clean_cond

        # Add noise to conditional frame
        # noisy_cond = None
        if self.cond_noise and isinstance(self.cond_noise, ListConfig):
            cond_noise = (self.cond_noise[0] + self.cond_noise[1] * torch.randn((1,))).exp()
            noisy_cond = noisy_cond + cond_noise * torch.randn_like(noisy_cond)
        elif self.cond_noise:
            noisy_cond = noisy_cond + self.cond_noise * torch.randn_like(noisy_cond)
            cond_noise = self.cond_noise
        else:
            cond_noise = None

        return clean_cond, noisy_cond, target, audio_frames, cond_noise

    def _get_indexes(self, video_filelist, audio_filelist):
        indexes = []
        self.og_shape = None
        for vid_file, audio_file in zip(video_filelist, audio_filelist):
            vr = decord.VideoReader(vid_file)
            if self.og_shape is None:
                self.og_shape = vr[0].shape[-2]
            len_video = len(vr)
            # Short videos
            if len_video < self.num_frames:
                continue
            else:
                possible_indexes = list(sliding_window(range(len_video), self.num_frames))[:: self.step]
                possible_indexes = list(map(lambda x: (x, vid_file, audio_file), possible_indexes))
                indexes.extend(possible_indexes)
        print("Indexes", len(indexes), "\n")
        return indexes

    def scale_and_crop(self, video):
        h, w = video.shape[-2], video.shape[-1]
        # scale shorter side to resolution

        if self.resize_size is not None:
            scale = self.resize_size / min(h, w)
            if h < w:
                target_size = (self.resize_size, math.ceil(w * scale))
            else:
                target_size = (math.ceil(h * scale), self.resize_size)
            video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False, antialias=True)

            # center crop
            h, w = video.shape[-2], video.shape[-1]
            w_start = (w - self.resize_size) // 2
            h_start = (h - self.resize_size) // 2
            video = video[:, :, h_start : h_start + self.resize_size, w_start : w_start + self.resize_size]
        return self.maybe_augment(video)

    def __getitem__(self, idx):
        clean_cond, noisy_cond, target, audio, cond_noise = self._get_frames_and_audio(idx)
        out_data = {}
        # out_data = {"cond": cond, "video": target, "audio": audio, "video_file": video_file}

        if audio is not None:
            out_data["audio_emb"] = audio

        if self.use_latent:
            input_key = "latents"
        else:
            input_key = "frames"
        out_data[input_key] = target
        if noisy_cond is not None:
            out_data["cond_frames"] = noisy_cond
        out_data["cond_frames_without_noise"] = clean_cond
        if cond_noise is not None:
            out_data["cond_aug"] = cond_noise
        out_data["motion_bucket_id"] = torch.tensor([self.motion_id])
        out_data["fps_id"] = torch.tensor([self.video_rate])
        out_data["num_video_frames"] = self.num_frames
        out_data["image_only_indicator"] = torch.zeros(self.num_frames)
        return out_data


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import cv2

    transform = transforms.Compose(transforms=[transforms.Resize((256, 256))])
    dataset = VideoDataset(
        "/vol/paramonos2/projects/antoni/datasets/mahnob/filelist_videos_val.txt", transform=transform, num_frames=25
    )
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))

    for i in range(10):
        print(dataset[i][0].shape, dataset[i][1].shape)

    image_identity = (dataset[idx][0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    image_other = (dataset[idx][1][:, -1].permute(1, 2, 0).numpy() + 1) / 2 * 255
    cv2.imwrite("image_identity.png", image_identity[:, :, ::-1])
    for i in range(25):
        image = (dataset[idx][1][:, i].permute(1, 2, 0).numpy() + 1) / 2 * 255
        cv2.imwrite(f"tmp_vid_dataset/image_{i}.png", image[:, :, ::-1])
