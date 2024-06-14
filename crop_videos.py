import os
import subprocess
from tqdm import tqdm

# Paths
dataset_dir = "/fsx/rs2517/data/HDTF/dataset_folder/HDTF_dataset"
input_dir = "/fsx/rs2517/data/HDTF/split_videos"  # Assuming split videos are stored here
output_dir = "./cropped_videos_original"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

prefixes = ["RD", "WDA", "WRA"]  # Possible prefixes


def crop_videos():
    for prefix in prefixes:
        crop_wh_path = os.path.join(dataset_dir, f"{prefix}_crop_wh.txt")
        print(crop_wh_path)
        if not os.path.isfile(crop_wh_path):
            print(f"No crop info file found for prefix {prefix}")
            continue
        with open(crop_wh_path, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Processing prefix {prefix}", total=len(lines)):
                parts = line.strip().split(" ")
                video_name_clip = f"{prefix}_" + parts[0].split(".")[0]
                min_width, width, min_height, height = map(int, parts[1:])
                input_path = os.path.join(input_dir, f"{video_name_clip}.mp4")
                
                if not os.path.isfile(input_path):
                    print(f"Input file {input_path} not found.")
                    continue
                output_path = os.path.join(output_dir, f"{video_name_clip}_cropped.mp4")
                cmd = f'ffmpeg -i {input_path} -filter:v "crop={width}:{height}:{min_width}:{min_height}, scale=512:512" {output_path}'
                subprocess.run(cmd, shell=True)
                quit()

crop_videos()
print("Video cropping and resizing complete.")
