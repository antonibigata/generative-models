import glob
import torch
from safetensors.torch import save_file
import argparse
from tqdm import tqdm
import os


def convert_to_safetensors(args):
    if args.filelist:
        with open(args.filelist, "r") as f:
            file_list = [line.strip() for line in f]
    else:
        file_list = glob.glob(args.glob_path)

    for file_path in tqdm(file_list, desc="Converting", total=len(file_list)):
        if file_path.endswith(".pt"):
            if args.only_beats and not file_path.endswith("_beats_emb.pt"):
                continue
            if not args.recompute and os.path.exists(file_path.replace(".pt", ".safetensors")):
                continue
            try:
                tensor = torch.load(file_path).contiguous()
                save_file({f"{args.name}": tensor}, file_path.replace(".pt", ".safetensors"))
                # print(f"Converted {file_path} to safetensors.")
            except Exception as e:
                print(f"Failed to convert {file_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert torch files to safetensors.")
    parser.add_argument("--glob_path", help="Path pattern to match torch files.")
    parser.add_argument("--filelist", help="Path to a file containing a list of torch files to convert.")
    parser.add_argument("--name", help="Name in the output file dict.", default="")
    parser.add_argument("--only_beats", action="store_true", help="Only convert files ending with _beats_emb")
    parser.add_argument("--recompute", action="store_true", help="Recompute existing safetensors files")
    args = parser.parse_args()

    if not args.glob_path and not args.filelist:
        parser.error("Either --glob_path or --filelist must be provided.")

    convert_to_safetensors(args)
