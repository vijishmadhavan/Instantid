from huggingface_hub import hf_hub_download
import requests
import os

# Function to download a file from Hugging Face
def download_from_huggingface(repo_id, filename, local_dir):
    if not os.path.exists(os.path.join(local_dir, filename)):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"Downloaded {filename} from {repo_id}")
    else:
        print(f"{filename} already exists in {local_dir}")

# Function to download a file from a URL
def download_from_url(file_url, save_dir):
    file_name = file_url.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)
    if not os.path.exists(file_path):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {file_name}")
        else:
            print(f"Failed to download {file_url}")
    else:
        print(f"{file_name} already exists in {save_dir}")

# Download models from Hugging Face
download_from_huggingface(
    "InstantX/InstantID", "ControlNetModel/config.json", "./checkpoints"
)
download_from_huggingface(
    "InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", "./checkpoints"
)
download_from_huggingface(
    "InstantX/InstantID", "ip-adapter.bin", "./checkpoints"
)
download_from_huggingface(
    "latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors", "./checkpoints"
)

files_to_download = [
    "https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/1k3d68.onnx",
    "https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/2d106det.onnx",
    "https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/genderage.onnx",
    "https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/glintr100.onnx",
    "https://huggingface.co/spaces/InstantX/InstantID/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx"
]

# Directory to save the downloaded files
save_dir = "models/antelopev2"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download each file from the URL
for file_url in files_to_download:
    download_from_url(file_url, save_dir)

print("Downloads completed.")
