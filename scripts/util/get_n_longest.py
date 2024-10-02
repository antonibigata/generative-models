import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import heapq
import argparse


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def get_dataset_from_path(video_path):
    # Extract dataset name from the video path
    path_parts = video_path.split("/")
    for part in path_parts:
        if part in ["HDTF", "CelebV-Text", "AA_processed", "CelebV_HQ"]:
            return part
    return "Unknown"


def process_videos(filelist):
    durations = defaultdict(list)
    for line in tqdm(filelist, desc="Processing videos"):
        video_path = line.strip()
        dataset = get_dataset_from_path(video_path)
        duration = get_video_duration(video_path)
        durations[dataset].append((duration, video_path))
    return durations


def get_n_longest_samples(durations, n):
    return heapq.nlargest(n, durations)


def main():
    parser = argparse.ArgumentParser(description="Find the n longest video samples from a specific dataset.")
    parser.add_argument("filelist", help="Path to the filelist containing video paths")
    parser.add_argument("target_dataset", help="Name of the target dataset")
    parser.add_argument(
        "-n", "--num_samples", type=int, default=10, help="Number of longest samples to retrieve (default: 10)"
    )
    args = parser.parse_args()

    filelist_path = args.filelist
    target_dataset = args.target_dataset
    n_longest = args.num_samples

    with open(filelist_path, "r") as f:
        filelist = f.readlines()

    durations = process_videos(filelist)
    longest_samples = get_n_longest_samples(durations[target_dataset], n_longest)

    output_file = f"{target_dataset}_longest_samples.txt"
    with open(output_file, "w") as f:
        f.write(f"Top {n_longest} longest samples for {target_dataset}:\n")
        print(f"Top {n_longest} longest samples for {target_dataset}:")
        for i, (duration, video_path) in enumerate(longest_samples, 1):
            f.write(f"{i}. Duration: {duration:.2f} seconds, Path: {video_path}\n")
            print(f"{i}. Duration: {duration:.2f} seconds, Path: {video_path}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
