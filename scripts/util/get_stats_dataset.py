import os
import cv2
from tqdm import tqdm
from collections import defaultdict


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        duration = frame_count / fps
    except ZeroDivisionError:
        duration = 0
    cap.release()
    return duration


def get_dataset_from_path(video_path):
    # Extract dataset name from the video path
    path_parts = video_path.split("/")
    for part in path_parts:
        if part in ["HDTF", "CelebV-Text", "AA_processed", "CelebV_HQ", "1000actors_nsv"]:
            return part
    return "Unknown"


def process_videos(filelist):
    durations = defaultdict(list)
    speakers = defaultdict(set)
    total_durations = defaultdict(float)  # To keep track of total duration in hours

    for line in tqdm(filelist, desc="Processing videos"):
        video_path = line.strip()
        dataset = get_dataset_from_path(video_path)
        duration = get_video_duration(video_path)

        # Convert duration to hours for comparison
        duration_hours = duration / 3600

        # Check conditions for AA_processed and 1000actors_nsv
        if dataset == "AA_processed" and total_durations[dataset] + duration_hours > 160:
            continue
        if dataset == "1000actors_nsv" and total_durations[dataset] + duration_hours > 30:
            continue

        durations[dataset].append(duration)
        total_durations[dataset] += duration_hours

        if dataset == "HDTF":
            speaker = "_".join(os.path.basename(video_path).split("_")[:2])
        else:
            speaker = os.path.basename(video_path).split("_")[0]
        speakers[dataset].add(speaker)

    return durations, speakers


def compute_stats(durations, speakers):
    return {
        "number_of_videos": len(durations),
        "number_of_speakers": len(speakers),
        "total_duration_seconds": sum(durations),
        "average_duration_seconds": sum(durations) / len(durations) if durations else 0,
    }


def main():
    filelist_path = "/data/home/antoni/datasets/filelist_celebhq_text_aa_hdtf_nsv.txt"

    with open(filelist_path, "r") as f:
        filelist = f.readlines()

    durations_by_dataset, speakers_by_dataset = process_videos(filelist)

    all_durations = []
    all_speakers = set()
    dataset_stats = {}

    output_file = "dataset_stats_updated.txt"
    with open(output_file, "w") as f:
        for dataset, durations in durations_by_dataset.items():
            all_durations.extend(durations)
            all_speakers.update(speakers_by_dataset[dataset])
            stats = compute_stats(durations, speakers_by_dataset[dataset])
            dataset_stats[dataset] = stats

            f.write(f"\nStats for {dataset}:\n")
            print(f"\nStats for {dataset}:")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
                print(f"{key}: {value}")

        total_stats = compute_stats(all_durations, all_speakers)

        f.write("\nTotal stats across all datasets:\n")
        print("\nTotal stats across all datasets:")
        for key, value in total_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
