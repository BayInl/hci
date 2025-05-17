# 图像搜索应用 - Five Stage Search Framework

这是一个基于 Gradio + CLIP + Upstash Vector 实现的图像搜索可视化应用，能够实现文本到图像和图像到图像的搜索功能。

## 功能特点

- **文本搜索图像**：用户输入文本描述，系统返回与描述最相似的图像
- **图像搜索图像**：用户上传一张图片，系统返回与该图片视觉上最相似的图像
- **收藏功能**：用户可以收藏感兴趣的图片（演示用户交互体验）
- **相似度排序**：搜索结果按相似度得分排序展示
- **详细信息展示**：为每个结果提供图像路径、类别和相似度分数

## 技术栈

- **Gradio**：用于构建交互式UI界面
- **CLIP**：OpenAI的多模态模型，用于理解图像和文本之间的语义关系
- **Upstash Vector**：用于向量存储和相似度搜索（本项目使用本地模拟实现）
- **PyTorch & TorchVision**：用于图像处理和模型加载

## 安装与使用

### 环境要求

- Python 3.10+
- CUDA（可选，用于GPU加速）

### 环境设置

1. 创建并激活conda环境
```bash
conda create -n py310 python=3.10
conda activate py310
```

2. 安装依赖
```bash
python -m venv .venv
source source .venv/bin/activate

pip install gradio torch torchvision ftfy regex tqdm upstash-vector pillow transformers sentence-transformers
pip install git+https://github.com/openai/CLIP.git
```

3. 下载示例数据集（如使用GroceryStoreDataset）
```bash
git clone https://github.com/marcusklasson/GroceryStoreDataset.git
```

### 运行预处理程序
```bash
python preprocess_data.py
```

### 运行主程序

```bash
python app.py
```

应用将在本地启动，并提供一个可以通过浏览器访问的URL。

## 使用指南

1. **文本搜索图像**：
   - 进入"文本搜索图像"标签页
   - 在文本框中输入描述（如"红苹果"、"香蕉"等）
   - 点击"搜索"按钮查看结果

2. **图像搜索图像**：
   - 进入"图像搜索图像"标签页
   - 上传一张图片
   - 点击"搜索"按钮查看与上传图片相似的图像

3. **收藏功能**：
   - 点击图片可以将其添加到收藏夹
   - 在"收藏夹"标签页查看已收藏的图片

## 项目说明

本项目实现了Five Stage Search Framework的要素：

1. **Query Input/Refinement**：通过文本输入框和图像上传功能实现
2. **Content Analysis**：使用CLIP模型对文本和图像进行语义编码
3. **Indexing**：将图像向量化并存储在向量数据库中
4. **Similarity Matching**：使用余弦相似度计算查询与库中图像的相似度
5. **Result Presentation**：通过Gallery组件以网格形式展示搜索结果

## 数据集

本项目使用[GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset)作为示例数据集，该数据集包含超市商品的图像。