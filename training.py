import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import IPAdapterModel  # 假设你修改后的模型类名称为 IPAdapterModel
from PIL import Image
import os

# ---------------------【MODIFIED】定义数据集类---------------------
class MyDataset(Dataset):
    """
    一个简单的 Dataset，返回每个样本的 PIL Image 和对应的文本描述。
    请将 image_paths 和 text_prompts 替换为你实际的数据路径和描述。
    """
    def __init__(self, image_paths, text_prompts, image_size=(512,512)):
        self.image_paths = image_paths
        self.text_prompts = text_prompts
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像（使用 PIL）
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # 调整大小
        img = img.resize(self.image_size)
        # 获取文本描述
        text = self.text_prompts[idx]
        return img, text
# ---------------------【MODIFIED】数据集定义结束---------------------

# ---------------------【MODIFIED】设置数据集和 DataLoader---------------------
# 示例图像路径和文本描述
image_paths = [
    "./assets/staff_1024.jpg",
    "./assets/pexels-freddy-barbosa.jpg",
    "./assets/pexels-mart-production.jpg"
]
text_prompts = [
    "three dogs sitting in white",
    "a woman with a dog sitting on the playground",
    "a man walking his dog"
]

train_dataset = MyDataset(image_paths, text_prompts, image_size=(512,512))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# ---------------------【MODIFIED】数据集和 DataLoader设置结束---------------------

# ---------------------【MODIFIED】加载预训练模型---------------------
# 预训练 checkpoint 路径为 "path/to/ip-adapter_sd15.bin"
# IPAdapterModel 支持参数 train_content_branch_only
model = IPAdapterModel.from_pretrained("path/to/ip-adapter_sd15.bin", train_content_branch_only=True)
model.train()  # 设置模型为训练模式
# ---------------------【MODIFIED】模型加载结束---------------------

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 训练参数
num_epochs = 50

# ---------------------【MODIFIED】训练循环---------------------
for epoch in range(num_epochs):
    for batch in train_loader:
        images, texts = batch  # images 是 PIL Image 列表，texts 是对应的描述文本
        optimizer.zero_grad()
        # 前向传播：假设模型支持直接输入图像和文本，并返回一个包含 loss 的输出
        outputs = model(images, texts)
        loss = outputs.loss
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
# ---------------------【MODIFIED】训练循环结束---------------------

print("Training complete!")