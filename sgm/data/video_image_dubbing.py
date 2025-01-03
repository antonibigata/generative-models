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
from safetensors.torch import load_file
from safetensors import safe_open

from sgm.data.data_utils import trim_pad_audio

from sgm.data.data_utils import (
    create_masks_from_landmarks_full_size,
    create_face_mask_from_landmarks,
    create_masks_from_landmarks_box,
)

torchaudio.set_audio_backend("sox_io")
decord.bridge.set_bridge("torch")


def do_skip(files_to_check, file):
    for skip in files_to_check:
        if skip in file:
            return True
    return False


# Similar to regular video dataset but trades flexibility for speed
class VideoDataset(Dataset):
    def __init__(
        self,
        filelist,
        resize_size=None,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        landmarks_folder=None,
        video_extension=".avi",
        audio_extension=".wav",
        audio_in_video=False,
        num_frames=16,
        additional_audio_frames=0,
        audio_rate=16000,
        latent_folder=None,
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
        virtual_increase=1,
        n_out_frames=1,
        is_xl=False,
        use_latent_condition=False,
        get_landmarks=False,
        skip_files=["id04974/5z4zNgCIe3c/00059"],
        change_file_proba=0.1,
        get_masks=False,
        only_predict_mouth=False,
        what_mask="full",
    ):
        self.audio_folder = audio_folder
        landmarks_folder = video_folder if landmarks_folder is None else landmarks_folder
        self.landmarks_folder = landmarks_folder
        self.num_frames = num_frames
        self.from_audio_embedding = from_audio_embedding
        self.allow_all_possible_permutations = allow_all_possible_permutations
        self.audio_emb_type = audio_emb_type
        self.get_difference_score = get_difference_score
        self.cond_noise = cond_noise
        self.motion_id = motion_id
        self.latent_condition = use_latent_condition
        self.is_xl = is_xl
        self.latent_folder = latent_folder if latent_folder is not None else video_folder
        self.audio_in_video = audio_in_video
        self.n_out_frames = n_out_frames
        self.change_file_proba = change_file_proba
        self.only_predict_mouth = only_predict_mouth
        self.what_mask = what_mask
        self.get_masks = get_masks

        self.filelist = []
        self.audio_filelist = []
        self.landmark_filelist = []
        with open(filelist, "r") as files:
            for f in files.readlines():
                f = f.rstrip()
                skip = do_skip(skip_files, f)
                if skip:
                    continue
                audio_path = f.replace(video_folder, audio_folder).replace(video_extension, audio_extension)
                # if not self.audio_in_video and not os.path.exists(audio_path):
                #     missing_audio += 1
                #     print("Missing audio file: ", audio_path)
                #     if missing_audio > max_missing_audio_files:
                #         raise FileNotFoundError(f"Missing more than {max_missing_audio_files} audio files")
                #     continue
                self.filelist += [f]
                self.audio_filelist += [audio_path]
                landmark_path = f.replace(video_folder, landmarks_folder).replace(video_extension, ".npy")
                self.landmark_filelist += [landmark_path]

        self.resize_size = resize_size
        self.scale_audio = scale_audio
        self.split_audio_to_frames = split_audio_to_frames
        self.use_latent = use_latent
        self.latent_type = latent_type
        self.latent_scale = latent_scale
        self.video_ext = video_extension
        self.video_folder = video_folder
        self.additional_audio_frames = additional_audio_frames
        self.virtual_increase = virtual_increase

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
            self._indexes = self._get_indexes(self.filelist, self.audio_filelist, self.landmark_filelist)
        else:
            # if self.get_landmarks:
            self._indexes = list(zip(self.filelist, self.audio_filelist, self.landmark_filelist))
            # else:
            #     self._indexes = list(zip(self.filelist, self.audio_filelist, [None] * len(self.filelist)))
        self.total_len = len(self._indexes)

        # Get metadata about video and audio
        # _, self.audio_rate = torchaudio.load(self.audio_filelist[0], channels_first=False)
        vr = decord.VideoReader(self.filelist[0])
        self.video_rate = math.ceil(vr.get_avg_fps())
        print(f"Video rate: {self.video_rate}")
        self.audio_rate = audio_rate
        a2v_ratio = self.video_rate / float(self.audio_rate)
        self.samples_per_frame = math.ceil(1 / a2v_ratio)
        self.curr_video = None
        self.curr_landmarks = None
        self.curr_audio = None
        self.curr_idx = None
        self.curr_latents = None

    def __len__(self):
        return len(self._indexes) * self.virtual_increase

    def _load_landmarks(self, original_size, target_size, indexes):
        landmarks = self.curr_landmarks[indexes, :][:, :, :2]
        if self.what_mask == "full":
            mask = create_masks_from_landmarks_full_size(landmarks, original_size[0], original_size[1], offset=-0.01)
        elif self.what_mask == "box":
            mask = create_masks_from_landmarks_box(landmarks, (original_size[0], original_size[1]), box_expand=0.0)
        else:
            mask = create_face_mask_from_landmarks(landmarks, original_size[0], original_size[1], mask_expand=0.05)
        # mask = create_masks_from_landmarks_full_size(landmarks, original_size[0], original_size[1], offset=-0.01)
        # Interpolate the mask to the target size
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode="nearest")
        return mask

    def _load_audio(self, filename, max_len_sec, start=None, return_all=False):
        if return_all:
            audio, sr = sf.read(filename, always_2d=True)
            audio = audio.T
            audio = torch.from_numpy(audio).float()
            audio = trim_pad_audio(audio, self.audio_rate, max_len_sec=max_len_sec)
            return audio[0]
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

    def get_audio_indexes(self, main_indexes, n_audio_frames, max_len):
        start, end = main_indexes[0], main_indexes[-1]
        # Get indexes for audio from both sides of the main index
        audio_ids = []
        # get audio embs from both sides of the GT frame
        audio_ids += [0] * max(n_audio_frames - start, 0)
        for i in range(max(start - n_audio_frames, 0), min(end + n_audio_frames + 1, max_len)):
            # for i in range(frame_ids[0], min(frame_ids[0] + self.n_audio_motion_embs + 1, n_frames)):
            audio_ids += [i]
        audio_ids += [max_len - 1] * max(end + n_audio_frames - max_len + 1, 0)
        return audio_ids, max(n_audio_frames - start, 0), max(end + n_audio_frames - max_len + 1, 0)

    def ensure_shape(self, tensors):
        target_length = self.samples_per_frame
        processed_tensors = []
        for tensor in tensors:
            current_length = tensor.shape[1]
            diff = current_length - target_length
            assert abs(diff) <= 5, f"Expected shape {target_length}, but got {current_length}"
            # print(f"Expected shape {target_length}, but got {current_length}")
            if diff < 0:
                # Calculate how much padding is needed
                padding_needed = target_length - current_length
                # Pad the tensor
                padded_tensor = F.pad(tensor, (0, padding_needed))
                processed_tensors.append(padded_tensor)
            elif diff > 0:
                # Trim the tensor
                trimmed_tensor = tensor[:, :target_length]
                processed_tensors.append(trimmed_tensor)
            else:
                # If it's already the correct size
                processed_tensors.append(tensor)
        return torch.cat(processed_tensors)

    def get_predict_index_with_buffer(self, max_frames, first_index):
        # Predict frame idx
        # Calculate valid range for the second index
        start = max(0, first_index - self.num_frames)
        end = min(max_frames, first_index + self.num_frames)
        valid_indices = list(range(0, start)) + list(range(end, max_frames - self.n_out_frames))
        # Select the second index from the valid range
        if valid_indices:
            second_index = np.random.choice(valid_indices)
        else:
            second_index = np.random.randint(0, max_frames)

        assert (
            second_index + self.n_out_frames <= max_frames
        ), f"Second index is {second_index} but max is {max_frames} and n_out_frames is {self.n_out_frames}"

        return [second_index] + [second_index + i for i in range(1, self.n_out_frames)]

    def convert_indexes(self, indexes_25fps, fps_from=25, fps_to=60):
        ratio = fps_to / fps_from
        indexes_60fps = [int(index * ratio) for index in indexes_25fps]
        return indexes_60fps

    def _get_frames_and_audio(self, idx):
        vr = self.curr_video
        if self.load_all_possible_indexes:
            indexes, _, _, _ = self._indexes[self.curr_idx]
            len_video = len(vr)
            if "AA_processed" in self.curr_file:
                len_video *= 25 / 60
                len_video = int(len_video)
        else:
            len_video = len(vr)
            if "AA_processed" in self.curr_file:
                len_video *= 25 / 60
                len_video = int(len_video)
            init_index = np.random.randint(0, len_video)
            predict_index = self.get_predict_index_with_buffer(len_video, init_index)
            indexes = [init_index, *predict_index]

        # Initial frame between 0 and len(video) - frame_space
        init_index = indexes[0]

        # Cond frame idx
        predict_index = indexes[1:]
        audio_indexes, left_copies, right_copies = self.get_audio_indexes(
            predict_index, self.additional_audio_frames, len(vr)
        )

        if "AA_processed" in self.curr_file:
            indexes = self.convert_indexes(indexes, fps_from=25, fps_to=60)
            init_index = indexes[0]
            predict_index = indexes[1:]
        # print(audio_indexes, indexes)

        audio_frames = None
        # raw_audio = None
        if self.audio_in_video:
            # raw_audio, _ = vr.get_batch(audio_indexes)
            vr = vr._AVReader__video_reader
            # raw_audio = rearrange(self.ensure_shape(raw_audio), "f s -> (f s)")

        if self.use_latent:
            # latent_file = video_file.replace(self.video_ext, f"_{self.latent_type}_512_latent.safetensors").replace(
            #     self.video_folder, self.latent_folder
            # )
            # frames = load_file(latent_file)["latents"]
            frames = self.curr_latents
            frame = frames[predict_index[0] : predict_index[-1] + 1, :, :, :]
            cond = frames[init_index : init_index + 1, :, :, :].squeeze(0)
            # except FileNotFoundError:
            #     frames = torch.load(video_file.replace(self.video_ext, "_latent.pt"))
            #     frame = frames[predict_index, :, :, :]
            #     cond = frames[init_index]
            frame = frame * self.latent_scale
            noisy_cond = cond * self.latent_scale
            clean_cond = vr[init_index].float()

        else:
            frame, clean_cond = vr.get_batch([*predict_index, init_index]).float()
            # clean_cond = vr[init_index].float()
        assert frame.shape[0] == self.n_out_frames, f"Frame shape is {frame.shape}, expected {self.n_out_frames}"

        or_w, or_h = clean_cond.shape[0], clean_cond.shape[1]
        # Maybe get mask from landmarks
        if self.get_masks:
            target_size = (
                (self.resize_size, self.resize_size)
                if not self.use_latent
                else (self.resize_size // 8, self.resize_size // 8)
            )
            masks = self._load_landmarks((or_w, or_h), target_size, predict_index).permute(1, 0, 2, 3)
            resized_masks = F.interpolate(masks, size=(noisy_cond.shape[-2], noisy_cond.shape[-1]), mode="nearest")
            if not self.only_predict_mouth:
                noisy_cond = noisy_cond * (1 - resized_masks)
        else:
            masks = torch.zeros_like(noisy_cond)
            masks[:, :, noisy_cond.shape[-2] // 2 :, :] = 1
            if not self.only_predict_mouth:
                noisy_cond = noisy_cond * (1 - masks)

        # if raw_audio is None:
        #     raw_audio = self._load_audio(audio_file, max_len_sec=len_video / self.video_rate, return_all=True)
        if not self.from_audio_embedding:
            assert False, "Not implemented"
            audio = raw_audio
            audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)[audio_indexes]
        else:
            # audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
            audio = self.curr_audio
            assert len(audio.get_shape()) == 3, f"Audio shape is {audio.shape}"
            audio_frames = audio[audio_indexes[0] : audio_indexes[-1] + 1, :]
            # Make copies of the first and last audio frames if needed
            if left_copies > 0:
                audio_frames = torch.cat([audio_frames[0][None, :]] * left_copies + [audio_frames])
            if right_copies > 0:
                audio_frames = torch.cat([audio_frames] + [audio_frames[-1][None, :]] * right_copies)
            assert audio_frames.shape[0] == (self.additional_audio_frames * 2 + 1)

        diff_score = None
        # if self.get_difference_score:
        #     if self.use_latent:
        #         print_once("Difference score for latent may not be accurate")
        #     diff_score = ssim(frame.numpy(), cond.numpy(), channel_axis=2, data_range=1)
        #     diff_score = ssim_to_bin(diff_score)

        # if audio_frames is None:
        #     audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)[audio_indexes]

        # if self.scale_audio:
        #     audio_frames = (audio_frames / audio_frames.max()) * 2 - 1

        if not self.use_latent:
            frame = rearrange(frame, "t h w c -> t c h w")
            clean_cond = rearrange(clean_cond, "h w c -> c h w")
            frame = self.scale_and_crop((frame / 255.0) * 2 - 1)
            clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)
            noisy_cond = clean_cond
        else:
            clean_cond = rearrange(clean_cond, "h w c -> c h w")
            clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)
            if not self.latent_condition:
                noisy_cond = clean_cond

        # target = frame.unsqueeze(1)
        target = rearrange(frame, "t c h w -> c t h w")

        if self.cond_noise and isinstance(self.cond_noise, ListConfig):
            cond_noise = (self.cond_noise[0] + self.cond_noise[1] * torch.randn((1,))).exp()
            noisy_cond = noisy_cond + cond_noise * torch.randn_like(noisy_cond)
        else:
            noisy_cond = noisy_cond + self.cond_noise * torch.randn_like(noisy_cond)
            cond_noise = self.cond_noise

        # print("audio_frames", audio_frames.shape)

        return clean_cond, noisy_cond, target, masks, audio_frames, diff_score, cond_noise

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

    def load_new_media(self):
        idx = np.random.randint(0, len(self._indexes))
        self.curr_idx = idx
        if self.load_all_possible_indexes:
            indexes, video_file, audio_file, land_file = self._indexes[idx]
            self.indexes_all_poss = indexes
            if self.audio_in_video:
                vr = decord.AVReader(video_file, sample_rate=self.audio_rate)
            else:
                vr = decord.VideoReader(video_file)
        else:
            video_file, audio_file, land_file = self._indexes[idx]
            if self.audio_in_video:
                vr = decord.AVReader(video_file, sample_rate=self.audio_rate)
            else:
                vr = decord.VideoReader(video_file)

        audio_file = audio_file.replace("_output_output", "")
        self.curr_file = video_file
        self.curr_video = vr

        audio_path_extra = f"_{self.audio_emb_type}_emb.safetensors"
        video_path_extra = f"_{self.latent_type}_512_latent.safetensors"
        allow_pickle = False
        if "AA_processed" in video_file:
            audio_path_extra = ".safetensors"
            video_path_extra = f"_{self.latent_type}_512_latent.safetensors"
            land_file = land_file.replace("_output_output", "_output_keypoints")
            allow_pickle = True

        self.curr_landmarks = np.load(land_file, allow_pickle=allow_pickle)

        with safe_open(audio_file.split(".")[0] + audio_path_extra, framework="pt") as f:
            self.curr_audio = f.get_slice("audio")
        # self.curr_audio = torch.load(audio_file.split(".")[0] + f"_{self.audio_emb_type}_emb.pt")
        if self.use_latent:
            latent_file = video_file.replace(self.video_ext, video_path_extra).replace(
                self.video_folder, self.latent_folder
            )
            with safe_open(latent_file, framework="pt") as f:
                self.curr_latents = f.get_slice("latents")
            # self.curr_latents = load_file(latent_file)["latents"]

    def __getitem__(self, idx, error=False):
        try:
            if error or (np.random.rand() > (1 - self.change_file_proba) or self.curr_video is None):
                self.load_new_media()
            clean_cond, noisy_cond, target, masks, audio, diff_score, cond_noise = self._get_frames_and_audio(
                idx % len(self._indexes)
            )
        except Exception as e:
            print(f"Error with index {idx}: {e}")
            # raise e
            return self.__getitem__(np.random.randint(0, len(self)), error=True)

        _, video_file, _ = self._indexes[self.curr_idx]

        # print("clean_cond", clean_cond.shape)
        # print("noisy_cond", noisy_cond.shape)
        # print("target", target.shape)
        # print("audio", audio.shape)

        out_data = {}
        # out_data = {"cond": cond, "video": target, "audio": audio, "video_file": video_file}

        out_data["video_file"] = video_file

        if diff_score is not None:
            out_data["diff_score"] = torch.tensor([diff_score])

        if audio is not None:
            out_data["audio_emb"] = audio
            # out_data["raw_audio"] = raw_audio

        if self.use_latent:
            input_key = "latents"
        else:
            input_key = "frames"
        out_data[input_key] = target
        if noisy_cond is not None:
            out_data["cond_frames"] = noisy_cond
        out_data["cond_frames_without_noise"] = clean_cond
        # out_data["cond_frames_without_noise"] = clean_cond
        if cond_noise is not None:
            out_data["cond_aug"] = cond_noise
        # out_data["motion_bucket_id"] = torch.tensor([self.motion_id])
        # out_data["fps_id"] = torch.tensor([self.video_rate - 1])
        # out_data["txt"] = "a portrait of a person"
        # out_data["num_video_frames"] = self.num_frames
        # out_data["image_only_indicator"] = torch.zeros(self.num_frames)
        out_data["num_video_frames"] = self.n_out_frames
        out_data["image_only_indicator"] = torch.zeros(self.n_out_frames)
        if self.only_predict_mouth:
            out_data["gt"] = target
        if masks is not None:
            out_data["masks"] = masks
        if self.is_xl:
            out_data["original_size_as_tuple"] = torch.tensor([self.resize_size, self.resize_size])
            out_data["crop_coords_top_left"] = torch.tensor([0, 0])
            out_data["target_size_as_tuple"] = torch.tensor([self.resize_size, self.resize_size])

        return out_data


def collate_fn(batch):
    try:
        out_data = {}
        for key in batch[0].keys():
            if key not in ["video_file", "num_video_frames", "cond_aug"]:
                out_data[key] = torch.stack([d[key] for d in batch])
            else:
                out_data[key] = [d[key] for d in batch]
        return out_data
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        for d in batch:
            print(d["video_file"])
            print(d["audio_emb"].shape)
            print(d["latents"].shape)
        raise e


if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose(transforms=[transforms.Resize((256, 256))])
    dataset = VideoDataset(
        "/vol/paramonos2/projects/antoni/datasets/mahnob/filelist_videos_val.txt", transform=transform, num_frames=25
    )
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))

    for i in range(len(dataset)):
        print(dataset[i][0].shape, dataset[i][1].shape)

    # image_identity = (dataset[idx][0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    # image_other = (dataset[idx][1][:, -1].permute(1, 2, 0).numpy() + 1) / 2 * 255
    # cv2.imwrite("image_identity.png", image_identity[:, :, ::-1])
    # for i in range(25):
    #     image = (dataset[idx][1][:, i].permute(1, 2, 0).numpy() + 1) / 2 * 255
    #     cv2.imwrite(f"tmp_vid_dataset/image_{i}.png", image[:, :, ::-1])
