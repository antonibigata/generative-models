#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 BASE_DIR SOURCE_FOLDER TARGET_FOLDER"
    exit 1
fi

# Assign arguments to variables
BASE_DIR="$1"
SOURCE_FOLDER="$2"
TARGET_FOLDER="$3"

# Find all source directories
find "$BASE_DIR" -type d -name "$SOURCE_FOLDER" | while read -r source_dir; do
    # Construct the corresponding target directory path
    target_dir="${source_dir/$SOURCE_FOLDER/$TARGET_FOLDER}"
    
    # Check if the target directory exists
    if [ -d "$target_dir" ]; then
        echo "Target directory already exists: $target_dir"
    else
        # Create the symlink
        ln -s "$source_dir" "$target_dir"
        echo "Created symlink: $target_dir -> $source_dir"
    fi
done