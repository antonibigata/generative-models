import glob
import torch
from safetensors.torch import save_file
import argparse
from tqdm import tqdm


def convert_to_safetensors(args):
    file_list = glob.glob(args.glob_path)
    for file_path in tqdm(file_list, desc="Converting", total=len(file_list)):
        if file_path.endswith(".pt"):
            try:
                tensor = torch.load(file_path)
                save_file({f"{args.name}": tensor}, file_path.replace(".pt", ".safetensors"))
                print(f"Converted {file_path} to safetensors.")
            except Exception as e:
                print(f"Failed to convert {file_path}: {str(e)}")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Convert torch files to safetensors.")
        parser.add_argument("glob_path", help="Path pattern to match torch files.")
        parser.add_argument("--name", help="Name in the output file dict.", default="")
        args = parser.parse_args()

        convert_to_safetensors(args)
