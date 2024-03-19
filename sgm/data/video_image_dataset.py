import os
import numpy as np
from functools import partial
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import math
import decord
from einops import rearrange
import torchaudio
import soundfile as sf
from omegaconf import ListConfig

# from src.utils.utils import print_once
from more_itertools import sliding_window
from itertools import permutations
from torchvision.transforms import RandomHorizontalFlip
from audiomentations import Compose, AddGaussianNoise, PitchShift
from skimage.metrics import structural_similarity as ssim

torchaudio.set_audio_backend("sox_io")
decord.bridge.set_bridge("torch")


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


# Similar to regular video dataset but trades flexibility for speed
class VideoDataset(Dataset):
    def __init__(
        self,
        filelist,
        resize_size=None,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        landmark_folder="lmks",
        video_extension=".avi",
        audio_extension=".wav",
        num_frames=16,
        additional_audio_frames=0,
        audio_rate=16000,
        max_missing_audio_files=10,
        from_audio_embedding=False,
        scale_audio=False,
        split_audio_to_frames=True,
        augment=False,
        augment_audio=False,
        use_latent=False,
        latent_type="stable",
        latent_scale=1,  # For backwards compatibility
        exclude_dataset=[],
        allow_all_possible_permutations=False,
        load_all_possible_indexes=False,
        audio_emb_type="wavlm",
        get_difference_score=False,
        cond_noise=[-3.0, 0.5],
        motion_id=255.0,
    ):
        self.audio_folder = audio_folder
        self.landmark_folder = landmark_folder
        self.num_frames = num_frames
        self.from_audio_embedding = from_audio_embedding
        self.allow_all_possible_permutations = allow_all_possible_permutations
        self.audio_emb_type = audio_emb_type
        self.get_difference_score = get_difference_score
        self.cond_noise = cond_noise
        self.motion_id = motion_id

        self.filelist = []
        self.audio_filelist = []
        missing_audio = 0
        with open(filelist, "r") as files:
            for f in files.readlines():
                f = f.rstrip()
                dataset_name = f.split("/")[-3]
                if dataset_name in exclude_dataset:
                    continue
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
        self.scale_audio = scale_audio
        self.split_audio_to_frames = split_audio_to_frames
        self.use_latent = use_latent
        self.latent_type = latent_type
        self.latent_scale = latent_scale
        self.video_ext = video_extension
        self.video_folder = video_folder
        self.additional_audio_frames = additional_audio_frames

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

        self.load_all_possible_indexes = load_all_possible_indexes
        if load_all_possible_indexes:
            self._indexes = self._get_indexes(self.filelist, self.audio_filelist)
        else:
            self._indexes = list(zip(self.filelist, self.audio_filelist))
        self.total_len = len(self._indexes)

        # Get metadata about video and audio
        # _, self.audio_rate = torchaudio.load(self.audio_filelist[0], channels_first=False)
        vr = decord.VideoReader(self.filelist[0])
        self.video_rate = math.ceil(vr.get_avg_fps())
        print(f"Video rate: {self.video_rate}")
        self.audio_rate = audio_rate
        a2v_ratio = self.video_rate / float(self.audio_rate)
        self.samples_per_frame = math.ceil(1 / a2v_ratio)

    def __len__(self):
        return len(self._indexes)

    def _load_audio(self, filename, max_len_sec, start=None, return_all=False):
        if return_all:
            audio, sr = sf.read(filename, always_2d=True)
            audio = audio.T
            audio = torch.from_numpy(audio).float()
            audio = trim_pad_audio(audio, self.audio_rate, max_len_sec=max_len_sec)
            return audio
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

    def get_audio_indexes(self, main_index, n_audio_frames, max_len):
        # Get indexes for audio from both sides of the main index
        audio_ids = []
        # get audio embs from both sides of the GT frame
        audio_ids += [0] * max(n_audio_frames - main_index, 0)
        for i in range(max(main_index - n_audio_frames, 0), min(main_index + n_audio_frames + 1, max_len)):
            # for i in range(frame_ids[0], min(frame_ids[0] + self.n_audio_motion_embs + 1, n_frames)):
            audio_ids += [i]
        audio_ids += [max_len - 1] * max(main_index + n_audio_frames - max_len + 1, 0)
        return audio_ids

    def _get_frames_and_audio(self, idx):
        if self.load_all_possible_indexes:
            indexes, video_file, audio_file = self._indexes[idx]
            vr = decord.VideoReader(video_file)
        else:
            video_file, audio_file = self._indexes[idx]
            vr = decord.VideoReader(video_file)
            len_video = len(vr)
            init_index = np.random.randint(0, len_video)
            predict_index = np.random.randint(0, len_video)
            indexes = [init_index, predict_index]

        # Initial frame between 0 and len(video) - frame_space
        init_index = indexes[0]

        # Cond frame idx
        predict_index = indexes[-1]
        audio_indexes = self.get_audio_indexes(predict_index, self.additional_audio_frames, len(vr))

        audio_frames = None

        if self.use_latent:
            try:
                frames = torch.load(video_file.replace(self.video_ext, f"_{self.latent_type}_latent.pt"))
                frame = frames[predict_index, :, :, :]
                cond = frames[init_index]
            except FileNotFoundError:
                frames = torch.load(video_file.replace(self.video_ext, "_latent.pt"))
                frame = frames[predict_index, :, :, :]
                cond = frames[init_index]
            frame = frame * self.latent_scale
            noisy_cond = cond * self.latent_scale
            clean_cond = vr[init_index].float()

            if not self.from_audio_embedding:
                audio = self._load_audio(
                    audio_file, max_len_sec=len(vr) / self.video_rate, start=None, return_all=True
                )
                audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)[audio_indexes]
            else:
                audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
                try:
                    audio_frames = audio[audio_indexes, :]
                except IndexError as e:
                    print(f"Index error for {audio_file}")
                    print(f"Audio shape: {audio.shape}")
                    print(f"Indexes: {audio_indexes}")
                    print(
                        f"Frames shape: \
                            {torch.load(video_file.replace(self.video_ext, f'_{self.latent_type}_latent.pt')).shape}"
                    )
                    raise e
        else:
            frame = vr[predict_index].float()
            clean_cond = vr[init_index].float()

            if not self.from_audio_embedding:
                audio = self._load_audio(
                    audio_file, max_len_sec=len(vr) / self.video_rate, start=None, return_all=True
                )
            else:
                audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
                audio_frames = audio[audio_indexes, :]

        diff_score = None
        if self.get_difference_score:
            if self.use_latent:
                print_once("Difference score for latent may not be accurate")
            diff_score = ssim(frame.numpy(), cond.numpy(), channel_axis=2, data_range=1)
            diff_score = ssim_to_bin(diff_score)

        if audio_frames is None:
            audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)[audio_indexes]

        if self.scale_audio:
            audio_frames = (audio_frames / audio_frames.max()) * 2 - 1

        frame = rearrange(frame, "h w c -> c h w")
        clean_cond = rearrange(clean_cond, "h w c -> c h w")

        if not self.use_latent:
            frame = self.scale_and_crop((frame / 255.0) * 2 - 1)
            clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)
            noisy_cond = clean_cond
        else:
            clean_cond = rearrange(clean_cond, "h w c -> c h w")
            clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)

        target = frame.unsqueeze(1)

        if self.cond_noise and isinstance(self.cond_noise, ListConfig):
            cond_noise = (self.cond_noise[0] + self.cond_noise[1] * torch.randn((1,))).exp()
            noisy_cond = noisy_cond + cond_noise * torch.randn_like(noisy_cond)
        else:
            noisy_cond = noisy_cond + self.cond_noise * torch.randn_like(noisy_cond)
            cond_noise = self.cond_noise

        return clean_cond, noisy_cond, target, audio_frames, diff_score, cond_noise

    def _get_indexes(self, video_filelist, audio_filelist):
        indexes = []
        self.og_shape = None
        for vid_file, audio_file in zip(video_filelist, audio_filelist):
            vr = decord.VideoReader(vid_file)
            if self.og_shape is None:
                self.og_shape = vr[0].shape[-2]
            len_video = len(vr)
            if self.allow_all_possible_permutations:
                possible_indexes = list(permutations(range(len_video), 2))
                possible_indexes = list(map(lambda x: (x, vid_file, audio_file), possible_indexes))
                indexes.extend(possible_indexes)
                continue
            # Short videos
            possible_indexes = list(sliding_window(range(len_video), self.num_frames))
            possible_indexes = list(map(lambda x: (x, vid_file, audio_file), possible_indexes))
            indexes.extend(possible_indexes)
        print("Indexes", len(indexes), "\n")
        return indexes

    def scale_and_crop(self, video):
        h, w = video.shape[-2], video.shape[-1]
        video = video.unsqueeze(0)  # (1, 3, h, w)
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
            video = video[:, h_start : h_start + self.resize_size, w_start : w_start + self.resize_size]  # noqa
        return self.maybe_augment(video).squeeze(0)

    def __getitem__(self, idx):
        clean_cond, noisy_cond, target, audio, diff_score, cond_noise = self._get_frames_and_audio(idx)

        out_data = {}
        # out_data = {"cond": cond, "video": target, "audio": audio, "video_file": video_file}

        if diff_score is not None:
            out_data["diff_score"] = torch.tensor([diff_score])

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
        out_data["fps_id"] = torch.tensor([self.video_rate - 1])
        # out_data["num_video_frames"] = self.num_frames
        # out_data["image_only_indicator"] = torch.zeros(self.num_frames)
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