import time
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import cv2
from PIL import Image
from ip_adapter import IPAdapterXL

base_model_path = r"./stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = r"./sdxl_models/image_encoder"
ip_ckpt = r"./sdxl_models/ip-adapter_sdxl.safetensors"
device = "cuda"

controlnet_path = r"./diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, use_safetensors=True, torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

ip_model = IPAdapterXL(
    pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"]
)

prompt = "a boy, a girl, masterpiece, best quality, high quality"
negative_prompt = "low contrast, text, watermark, lowres, low quality, worst quality, deformed, glitch, noisy, blurry"
seed = 42
steps = 30
guidance = 3
multi_subject_emb = 2

content_path = "./assets/snow.jpeg"
content_image = Image.open(content_path).resize((512, 512))

# canny (from original file, consistent with your current code)
input_image = cv2.imread(content_path)
detected_map = cv2.Canny(input_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

edge_density = float((detected_map > 0).mean())
conflict_score = max(0.0, min(1.0, edge_density * 2.0))

base_style_scale = 1.0
base_control_scale = 0.8
style_scale = base_style_scale * (1.0 - 0.6 * conflict_score)
control_scale = base_control_scale + 0.3 * conflict_score

print(f"[DEBUG] edge_density={edge_density:.4f}, conflict_score={conflict_score:.4f}")
print(f"[DEBUG] style_scale={style_scale:.4f}, control_scale={control_scale:.4f}")

styles = [
    "./assets/style1.jpg",
    "./assets/style2.jpg",
    "./assets/style3.png",
]

for i, style_path in enumerate(styles, start=1):
    style_img = Image.open(style_path).resize((512, 512))

    # Ours record (you already have images1~3_ours, but this will regenerate & log)
    out_name = f"image{i}_ours"
    t0 = time.time()
    images = ip_model.generate(
        pil_image=style_img,
        content_image=content_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scale=style_scale,
        guidance_scale=guidance,
        num_samples=1,
        num_inference_steps=steps,
        seed=seed,
        image=canny_map,
        controlnet_conditioning_scale=control_scale,
        content_early_stop_th=1e-3,
        multi_subject_emb=multi_subject_emb,

        # record switch + metadata
        save_record=True,
        record_path=f"{out_name}.json",
        output_name=out_name,
        style_path=style_path,
        content_path=content_path,
        edge_density=edge_density,
        conflict_score=conflict_score,
        style_scale_used=style_scale,
        control_scale_used=control_scale,
    )
    images[0].save(f"{out_name}.png")
    print(f"[TIME] {out_name}: {time.time()-t0:.2f}s")