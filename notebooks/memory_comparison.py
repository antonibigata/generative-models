from memory_profiler import profile

# Assuming these functions are defined to work with your specific file format
from safetensors import safe_open
from safetensors.torch import load_file

path = "/fsx/rs2517/data/HDTF/cropped_videos_original/WRA_ToddYoung_000_video_512_latent.safetensors"


@profile
def method1():
    tensors = {}
    with safe_open(path, framework="pt") as f:
        tensor_slice = f.get_slice("latents")
    print("Method 1: Tensor slice loaded.")


@profile
def method2():
    tensor = load_file(path)["latents"]
    print("Method 2: Entire tensor loaded.")


if __name__ == "__main__":
    method1()
    method2()
