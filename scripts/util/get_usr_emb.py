import os
import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm


def process_audio_embeddings(file_list_path):
    with open(file_list_path, "r") as file:
        audio_files = file.read().splitlines()

    # Randomize the order of the audio files
    np.random.shuffle(audio_files)

    for audio_file in tqdm(audio_files, desc="Processing audio embeddings", total=len(audio_files)):
        audio_file = audio_file.replace("audio", "audio_emb").replace(".wav", ".npy")
        output_path = audio_file.replace(".npy", "_usr_emb.safetensors")

        if os.path.exists(output_path):
            continue

        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            continue

        # Initialize a list to store embeddings from all folders
        all_embeddings = [np.load(audio_file)]

        # Process embeddings from audio_emb and audio_emb_0 to audio_emb_23
        for i in range(24):
            emb_path = audio_file.replace("audio_emb", f"audio_emb_{i}")

            all_embeddings.append(np.load(emb_path))

        if all_embeddings:
            # Calculate the mean of all embeddings
            mean_embedding = np.mean(all_embeddings, axis=0)

            # Convert to torch tensor
            mean_embedding_tensor = torch.from_numpy(mean_embedding)

            # Save the mean embedding as a safetensors file

            save_file({"audio": mean_embedding_tensor}, output_path)
            # print(f"Saved mean embedding to {output_path}")
        else:
            print(f"No embeddings found for {audio_file}")


if __name__ == "__main__":
    file_list_path = (
        "/data/home/antoni/datasets/filelist_audio_all.txt"  # Replace with the actual path to your file list
    )
    process_audio_embeddings(file_list_path)
