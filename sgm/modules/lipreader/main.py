import os
import glob
import torchaudio

import torch
import torchvision
from lightnining import ModelModule
from argparse import ArgumentParser
from datamodule.transforms import VideoTransform

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def get_lightning_module(args):
    modelmodule = ModelModule()
    modelmodule.model.load_state_dict(
        torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    )
    return modelmodule


class VSR_DataLoader(torch.nn.Module):
    def __init__(self, detector="retinaface"):
        super().__init__()
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector(device="cuda:0")
            self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.landmarks_detector(data_filename)
        video = self.load_video(data_filename)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)
        return video

    def forward_preprocessed(self, data_filename):
        video = self.load_video(data_filename)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video, t = self.video_transform(video)
        return video, t


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="../models/vsr_trlrs3_base.max400.pth",
        help="The absolute path to the pre-trained checkpoint file, (Default: ../models/vsr_trlrs3_base.max400.pth)",
    )
    parser.add_argument(
        "--extract-position",
        type=str,
        default=None,
        choices=[None, "frontend", "encoder"],
        help="Position of feature extraction",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="preprocessed",
        choices=["preprocessed", "original"],
        help="Type of input, (Default: preprocessed)",
    )
    return parser.parse_args()


def cli_main():
    args = parse_args()
    model = get_lightning_module(args)
    model.eval()
    data_loader = VSR_DataLoader()

    # Initialize variables for benchmark evaluation
    total_edit_distance = 0.
    total_length = 0.

    # Determine the source filenames based on the input type
    if args.input_type == "preprocessed":
        preprocessed_dir = "../preprocessed_data"
        source_filenames = sorted(glob.glob(f"{preprocessed_dir}/lrs3_video_seg24s/test/*/*.mp4"))
    elif args.input_type == "original":
        original_dir = "/vol/hci2/Databases/audio-visual/LRS3"
        source_filenames = sorted(glob.glob(f"{original_dir}/test/*/*.mp4"))

    # Process each source file
    for source_index, source_file in enumerate(source_filenames):
        # Handle preprocessed input type
        if args.input_type == "preprocessed":
            text_file_path = source_file.replace("video", "text")[:-4] + ".txt"
            target = open(text_file_path).read().splitlines()[0]
            preprocessed_video, t = data_loader.forward_preprocessed(source_file)
        # Handle original input type
        elif args.input_type == "original":
            text_line_list = open(source_file[:-4] + ".txt").read().splitlines()[0].split(" ")
            target = " ".join(text_line_list[2:])
            preprocessed_video, t = data_loader(source_file)

        with torch.no_grad():
            if args.extract_position:
                output = model(preprocessed_video, extract_position=args.extract_position)
                print(f"Feature size: {output.size()}")
            else:
                pred = model(preprocessed_video)
                total_edit_distance += compute_word_level_distance(target, pred)
                total_length += len(target.split())
                print(f"Target: {target}\nPrediction: {pred}\n")
                print(f"Processed element {source_index}; WER: {total_edit_distance / total_length}\n")


if __name__ == "__main__":
    cli_main()
