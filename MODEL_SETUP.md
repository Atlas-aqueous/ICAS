# Model Setup

This repository does not include all large pretrained model weights. Large model files are excluded from GitHub because of file-size limits, repository-size concerns, and Git LFS quota limitations.

To reproduce the project, download the required public model files from their original sources and place them in the local paths listed below.

## 1. SDXL Base Model

Source:
stabilityai/stable-diffusion-xl-base-1.0

Local path:
./stabilityai/stable-diffusion-xl-base-1.0/

Expected structure:
./stabilityai/stable-diffusion-xl-base-1.0/model_index.json
./stabilityai/stable-diffusion-xl-base-1.0/scheduler/
./stabilityai/stable-diffusion-xl-base-1.0/tokenizer/
./stabilityai/stable-diffusion-xl-base-1.0/tokenizer_2/
./stabilityai/stable-diffusion-xl-base-1.0/text_encoder/
./stabilityai/stable-diffusion-xl-base-1.0/text_encoder_2/
./stabilityai/stable-diffusion-xl-base-1.0/unet/
./stabilityai/stable-diffusion-xl-base-1.0/vae/
./stabilityai/stable-diffusion-xl-base-1.0/vae_1_0/
./stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/
./stabilityai/stable-diffusion-xl-base-1.0/vae_encoder/

## 2. ControlNet Canny for SDXL

Source:
diffusers/controlnet-canny-sdxl-1.0

Local path:
./diffusers/controlnet-canny-sdxl-1.0/

Expected structure:
./diffusers/controlnet-canny-sdxl-1.0/config.json
./diffusers/controlnet-canny-sdxl-1.0/diffusion_pytorch_model.safetensors
./diffusers/controlnet-canny-sdxl-1.0/diffusion_pytorch_model.bin
./diffusers/controlnet-canny-sdxl-1.0/diffusion_pytorch_model.fp16.safetensors
./diffusers/controlnet-canny-sdxl-1.0/diffusion_pytorch_model.fp16.bin

The demo and inference examples use the standard .safetensors weights.

## 3. IP-Adapter Weights

Source:
h94/IP-Adapter

Place SDXL weights at:
./sdxl_models/

Typical SDXL files:
./sdxl_models/image_encoder/
./sdxl_models/ip-adapter_sdxl.bin
./sdxl_models/ip-adapter_sdxl.safetensors
./sdxl_models/ip-adapter_sdxl_vit-h.bin
./sdxl_models/ip-adapter_sdxl_vit-h.safetensors
./sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
./sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
./sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin
./sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors

Place SD1.5 weights at:
./models/

Typical SD1.5 files:
./models/ip-adapter_sd15.bin
./models/ip-adapter_sd15.safetensors
./models/ip-adapter_sd15_light.bin
./models/ip-adapter_sd15_light.safetensors
./models/ip-adapter_sd15_light_v11.bin
./models/ip-adapter_sd15_vit-G.bin
./models/ip-adapter_sd15_vit-G.safetensors
./models/ip-adapter-plus_sd15.bin
./models/ip-adapter-plus_sd15.safetensors
./models/ip-adapter-plus-face_sd15.bin
./models/ip-adapter-plus-face_sd15.safetensors
./models/ip-adapter-full-face_sd15.bin
./models/ip-adapter-full-face_sd15.safetensors

## 4. Local Inventory Files

The following files record the model layout of the current server environment:

model_files_with_size.txt
model_files_sha256.txt
model_path_references.txt
public_sdxl_files.txt
public_controlnet_files.txt
public_ipadapter_files.txt
need_manual_check_files.txt
large_files_over_500M.txt
large_files_over_500M_with_size.txt

Meanings:

model_files_with_size.txt: all local model-like files and their sizes.
model_files_sha256.txt: SHA256 checksums for local model-like files.
model_path_references.txt: model paths referenced by Python, Markdown, or notebook files.
public_sdxl_files.txt: local files classified as SDXL Base files.
public_controlnet_files.txt: local files classified as ControlNet Canny SDXL files.
public_ipadapter_files.txt: local files classified as IP-Adapter files.
need_manual_check_files.txt: files not matched to the known public model directories.
large_files_over_500M.txt: local files larger than 500MB.
large_files_over_500M_with_size.txt: local files larger than 500MB with file sizes.

Before deleting or rebuilding the server environment, check need_manual_check_files.txt. If it is empty, the large model files are likely reproducible from public model sources. If it contains custom checkpoints, manually back them up.

## 5. Recommended Download Commands

Install Hugging Face Hub CLI:

pip install -U "huggingface_hub<1.0"

Download SDXL Base:

huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./stabilityai/stable-diffusion-xl-base-1.0 --local-dir-use-symlinks False

Download ControlNet Canny SDXL:

huggingface-cli download diffusers/controlnet-canny-sdxl-1.0 --local-dir ./diffusers/controlnet-canny-sdxl-1.0 --local-dir-use-symlinks False

Download IP-Adapter:

huggingface-cli download h94/IP-Adapter --local-dir ./IP-Adapter-weights --local-dir-use-symlinks False

After downloading h94/IP-Adapter, copy the required SDXL and SD1.5 weights into:

./sdxl_models/
./models/

## 6. Reproduction Notes

If a script reports a missing model file, first inspect the path in the script, then compare it with model_path_references.txt.

If the missing file belongs to SDXL Base, download it from:
stabilityai/stable-diffusion-xl-base-1.0

If the missing file belongs to ControlNet Canny SDXL, download it from:
diffusers/controlnet-canny-sdxl-1.0

If the missing file belongs to IP-Adapter, download it from:
h94/IP-Adapter

Do not upload full SDXL, ControlNet, or IP-Adapter model directories to ordinary Git history. Use Git LFS only for selected files when necessary, and prefer external storage for full environment backup.
