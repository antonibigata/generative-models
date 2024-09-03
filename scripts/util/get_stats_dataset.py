import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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
        durations[dataset].append(duration)
    return durations


def compute_stats(durations):
    return {
        "count": len(durations),
        "total_duration_seconds": sum(durations),
        "total_duration_minutes": sum(durations) / 60,
        "total_duration_hours": sum(durations) / 3600,
        "mean_duration_seconds": np.mean(durations),
        "mean_duration_minutes": np.mean(durations) / 60,
        "mean_duration_hours": np.mean(durations) / 3600,
        "median_duration_seconds": np.median(durations),
        "median_duration_minutes": np.median(durations) / 60,
        "median_duration_hours": np.median(durations) / 3600,
        "min_duration_seconds": min(durations),
        "min_duration_minutes": min(durations) / 60,
        "min_duration_hours": min(durations) / 3600,
        "max_duration_seconds": max(durations),
        "max_duration_minutes": max(durations) / 60,
        "max_duration_hours": max(durations) / 3600,
    }


def main():
    filelist_path = "/data/home/antoni/datasets/filelist_celebhq_text_aa_hdtf.txt"

    with open(filelist_path, "r") as f:
        filelist = f.readlines()

    durations_by_dataset = process_videos(filelist)

    all_durations = []
    dataset_stats = {}

    output_file = "dataset_stats.txt"
    with open(output_file, "w") as f:
        for dataset, durations in durations_by_dataset.items():
            all_durations.extend(durations)
            stats = compute_stats(durations)
            dataset_stats[dataset] = stats

            f.write(f"\nStats for {dataset}:\n")
            print(f"\nStats for {dataset}:")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
                print(f"{key}: {value}")

        total_stats = compute_stats(all_durations)

        f.write("\nTotal stats across all datasets:\n")
        print("\nTotal stats across all datasets:")
        for key, value in total_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
