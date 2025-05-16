import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from upstash_vector import Index
import uuid
from config import UPSTASH_VECTOR_URL, UPSTASH_VECTOR_TOKEN, IMAGE_DATASET_PATH, model_name


if torch.backends.mps.is_available():
    print("MPS backend is available.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA backend is available.")
    device = torch.device("cuda")
else:
    print("No GPU backend is available.")
    device = torch.device("cpu")

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print("CLIP model and processor loaded.")

# 2. 初始化 Upstash Vector 索引
try:
    index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
    print("Successfully connected to Upstash Vector.")
except Exception as e:
    print(f"Error connecting to Upstash Vector: {e}")
    exit()

# 3. 处理并索引图片
all_image_files = []
for root, _, files in os.walk(IMAGE_DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            all_image_files.append(os.path.join(root, file))
print(f"Found {len(all_image_files)} images to process.")

batch_size = 128
vectors_to_upsert = []

for i, image_path in enumerate(all_image_files):
    try:
        image_id = str(uuid.uuid4()) # 为每个图片生成一个唯一的ID
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化特征
        # 将图片路径作为元数据存储，方便后续检索显示
        vector_data = (
            image_id,
            image_features.cpu().numpy()[0].tolist(),
            {"image_path": image_path, "filename": os.path.basename(image_path)}
        )
        vectors_to_upsert.append(vector_data)
        if len(vectors_to_upsert) >= batch_size or (i + 1) == len(all_image_files):
            print(f"Upserting batch of {len(vectors_to_upsert)} vectors to Upstash... (Image {i+1}/{len(all_image_files)})")
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []
            print(f"Batch upserted. Processed {i+1}/{len(all_image_files)} images.")
    except Exception as e:
        print(f"Error processing or upserting {image_path}: {e}")

print("-------------------------------------------------")
print("图片预处理和索引完成！")
print(f"总共处理了 {len(all_image_files)} 张图片。")
print("您现在可以运行主应用 app.py 了。")
print("-------------------------------------------------")
