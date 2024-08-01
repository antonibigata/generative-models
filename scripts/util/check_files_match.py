import os

# Define the file paths
FILE_LIST = "/fsx/behavioural_computing_data/face_generation_data/AA_processed/file_list_train_internal.txt"
NEW_FILE_LIST = "/fsx/behavioural_computing_data/face_generation_data/AA_processed/file_list_existing.txt"

# Define the extensions and folders to check
EXTENSIONS = ["mp4", "pt", "safetensors"]  # Replace with actual extensions
FOLDERS = {"mp4": "video_crop", "pt": "audio_emb", "safetensors": "video_crop_emb"}


# Function to remove '_output_output' from the path if it exists
def sanitize_path(path):
    return path.replace("_output_output", "").replace("video_aligned_512", "")


# Function to check if all required files exist
def check_files(base_path, base_name):
    for ext, folder in FOLDERS.items():
        if ext == "safetensors":
            paths_to_check = [
                f"{base_path}/{folder}/{base_name}_output_output_video_512_latent.{ext}",
                f"{base_path}/{folder}/{base_name}.{ext}",
            ]
        else:
            paths_to_check = [
                f"{base_path}/{folder}/{base_name}.{ext}",
                f"{base_path}/{folder}/{base_name}_output_output.{ext}",
            ]
        if not any(os.path.isfile(path) for path in paths_to_check):
            return False

    return True


# Initialize missing files count
missing_count = 0

# Open the new file list for writing
with open(NEW_FILE_LIST, "w") as new_file_list:
    # Read the existing file list
    with open(FILE_LIST, "r") as file_list:
        for file in file_list:
            file = file.strip()
            sanitized_file = sanitize_path(file)
            base_path = "/fsx/behavioural_computing_data/face_generation_data/AA_processed/" + os.path.dirname(
                sanitized_file
            )
            base_name = os.path.splitext(os.path.basename(sanitized_file))[0]

            # Check if all required files exist
            if check_files(base_path, base_name):
                new_file_list.write(f"{file}\n")
            else:
                missing_count += 1

# Print the number of missing files
print(f"Number of missing files: {missing_count}")
print(f"Number of existing files: {sum(1 for _ in open(NEW_FILE_LIST))}")
