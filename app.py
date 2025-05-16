import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from upstash_vector import Index
import os
import shutil

from config import UPSTASH_VECTOR_URL, UPSTASH_VECTOR_TOKEN, IMAGE_DATASET_PATH, DOWNLOAD_DIR, model_name

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

if torch.backends.mps.is_available():
    print("MPS backend is available.")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA backend is available.")
    device = torch.device("cuda")
else:
    print("No GPU backend is available.")
    device = torch.device("cpu")

try:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    print("CLIP model and processor loaded.")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    MODEL_LOADED = False


# 2. 初始化 Upstash Vector 索引
try:
    index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
    print("Successfully connected to Upstash Vector.")
    UPSTASH_LOADED = True
except Exception as e:
    print(f"Error connecting to Upstash Vector: {e}")
    UPSTASH_LOADED = False

# --- 核心功能 ---
def encode_text(text_query):
    if not MODEL_LOADED: 
        return None
    inputs = processor(text=text_query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True) # 归一化
    return text_features.cpu().numpy()[0].tolist()

def encode_image(image_pil):
    if not MODEL_LOADED: 
        return None
    if image_pil is None: 
        return None
    image_pil_rgb = image_pil.convert("RGB")
    inputs = processor(images=image_pil_rgb, return_tensors="pt",padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
        image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化
    return image_features.cpu().numpy()[0].tolist()

def search_vectors(query_vector, top_k=12):
    if not UPSTASH_LOADED or query_vector is None:
        return [], "Upstash Vector 未连接或查询向量为空。"
    
    try:
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True, # 确保获取元数据（包含 image_path）
            include_vectors=False # 通常不需要在结果中返回向量本身
        )
        # results 是一个包含 Match 对象的列表
        # Match 对象有 id, score, metadata 属性
        image_paths_scores = []
        for match in results:
            if match.metadata and 'image_path' in match.metadata:
                # 检查图片文件是否存在
                if os.path.exists(match.metadata['image_path']):
                    image_paths_scores.append((match.metadata['image_path'], match.score))
                else:
                    print(f"警告: 数据库中的图片路径无效或文件不存在: {match.metadata['image_path']}")
            else:
                print(f"警告: 结果 {match.id} 缺少 image_path元数据。")
        if not image_paths_scores:
            return [], "未找到有效的图片路径，请检查预处理步骤和Upstash中的元数据。"
        return image_paths_scores, f"找到 {len(image_paths_scores)} 张相似图片。"
    except Exception as e:
        return [], f"搜索失败: {e}"
    
# --- Gradio 应用逻辑 ---
# 状态变量，用于存储收藏夹
# Gradio 的 State 用于在会话中保持数据
initial_favorites = []
def handle_text_search(text_query, favorites_list_state):
    if not MODEL_LOADED or not UPSTASH_LOADED:
        return [], "模型或数据库未加载。", favorites_list_state
    if not text_query.strip():
        return [], "请输入搜索文本。", favorites_list_state
    print(f"Formulation: Text query = '{text_query}'") # Stage 1
    query_vector = encode_text(text_query)
    print("Initiation of action: Text search initiated.") # Stage 2
    results, message = search_vectors(query_vector)
    print(f"Review of results: Found {len(results)} images for text query.") # Stage 3
    # results is a list of (image_path, score)
    display_images = [res[0] for res in results]
    return display_images, message, favorites_list_state

def handle_image_search(image_pil, favorites_list_state):
    if not MODEL_LOADED or not UPSTASH_LOADED:
        return [], "模型或数据库未加载。", favorites_list_state
    if image_pil is None:
        return [], "请上传一张图片。", favorites_list_state
    print("Formulation: Image query provided.") # Stage 1
    query_vector = encode_image(image_pil)
    print("Initiation of action: Image search initiated.") # Stage 2
    results, message = search_vectors(query_vector)
    print(f"Review of results: Found {len(results)} images for image query.") # Stage 3
    display_images = [res[0] for res in results]
    return display_images, message, favorites_list_state

def add_to_favorites(image_path, favorites_list_state):
    if image_path and image_path not in favorites_list_state:
        new_favorites_list = favorites_list_state + [image_path]
        print(f"Use: Image '{os.path.basename(image_path)}' added to favorites.") # Stage 5
        return new_favorites_list, f"已收藏: {os.path.basename(image_path)}", new_favorites_list
    msg = f"已在收藏夹中或路径无效。" if image_path in favorites_list_state else "未选择图片。"
    return favorites_list_state, msg, favorites_list_state

def clear_favorites(favorites_list_state):
    print("Favorites cleared.")
    return [], "收藏夹已清空。", []

def view_favorites(favorites_list_state):
    if not favorites_list_state:
        return [], "收藏夹是空的。", favorites_list_state
    print(f"Review of results (favorites): Displaying {len(favorites_list_state)} favorite images.") # Stage3 (for favorites)
    return favorites_list_state, f"共 {len(favorites_list_state)} 张收藏。", favorites_list_state

# 模拟下载功能：Gradio 本身不直接处理文件下载到用户任意位置，
# 这里我们将图片复制到一个临时文件夹，并返回该路径，用户可以从那里获取。
# 更健壮的下载通常需要一个简单的HTTP服务器或Gradio的文件组件。

def download_image_action(image_path):
    if image_path and os.path.exists(image_path):
        try:
            filename = os.path.basename(image_path)
            dest_path = os.path.join(DOWNLOAD_DIR, filename)
            shutil.copy2(image_path, dest_path) # 复制图片
            print(f"Use: Image '{filename}' prepared for download at '{dest_path}'.") # Stage 5
            # gr.File 期望一个文件路径或BytesIO对象列表
            return gr.File(value=dest_path, label=f"下载: {filename}"), f"图片 '{filename}' 已准备好，请通过下面的链接下载。"
        except Exception as e:
            return None, f"下载失败: {str(e)}"
    return None, "无效的图片路径或文件不存在。"

# Refinement: 当用户点击结果中的图片时，使用该图片进行新的搜索
def search_by_selected_image(evt: gr.SelectData, favorites_list_state):
    # evt.value 是被选中图片的路径
    selected_image_path = evt.value
    print(f"Refinement: Clicked on result image '{os.path.basename(selected_image_path)}' to start new search.") # Stage 4
    if selected_image_path and os.path.exists(selected_image_path):
        try:
            pil_image = Image.open(selected_image_path)
            return handle_image_search(pil_image, favorites_list_state)
        except Exception as e:
            return [], f"无法加载选中图片: {str(e)}", favorites_list_state
    return [], "选中的图片路径无效。", favorites_list_state

# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼 图像搜索系统 (CLIP + Upstash Vector + Gradio)")
    gr.Markdown(
        "实现 Five Stage of Search Framework: "
        "1. **Formulation**: 使用文本或图片输入您的搜索需求. "
        "2. **Initiation**: 点击搜索按钮. ""3. **Review**: 查看下方展示的搜索结果. "
        "4. **Refinement**: 点击结果中的任一图片，以该图片为基础进行新的搜索. "
        "5. **Use**: 收藏或下载您感兴趣的图片."
    )
    if not MODEL_LOADED or not UPSTASH_LOADED:
        gr.Warning("错误：CLIP 模型或 Upstash Vector 未能成功加载。请检查控制台输出和您的配置（UPSTASH_VECTOR_URL, UPSTASH_VECTOR_TOKEN）。应用功能将受限。")
    # 状态变量，用于存储收藏夹列表
    favorites_list_state = gr.State(value=initial_favorites)
    # 状态变量，用于存储当前显示的图片，以便收藏和下载
    current_gallery_images = gr.State(value=[])
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Formulation & 2. Initiation")
            text_query_input = gr.Textbox(label="按文本搜索 (Text Query)", placeholder="例如：一只在草地上奔跑的狗")
            text_search_button = gr.Button("🔍 按文本搜索", variant="primary")
            image_query_input = gr.Image(type="pil", label="按图片搜索 (Image Query)")
            image_search_button = gr.Button("🖼 按图片搜索", variant="primary")
            gr.Markdown("### 5. Use: 收藏夹管理")
            view_favorites_button = gr.Button("⭐ 查看收藏夹")
            clear_favorites_button = gr.Button("🗑 清空收藏夹")
            favorites_status_message = gr.Textbox(label="收藏夹状态", interactive=False)
        with gr.Column(scale=3):
            gr.Markdown("### 3. Review, 4. Refinement & 5. Use")
            search_status_message = gr.Textbox(label="搜索状态", interactive=False)
            results_gallery = gr.Gallery(
                label="搜索结果 (点击图片可基于该图再次搜索，或进行收藏/下载)",
                show_label=True,
                columns=[4], # 每行显示4张图片
                object_fit="contain",
                height="auto",
                preview=True
            )
            # 用于显示收藏和下载按钮
            with gr.Row(visible=False) as action_buttons_row: # 初始隐藏，选中图片后显示
                selected_image_path_for_actions = gr.Textbox(label="当前选中图片路径",
                interactive=False, visible=False) # 存储选中图片路径
                add_to_favorites_button = gr.Button("❤️ 收藏选中图片")
                download_selected_button = gr.Button("⬇️ 下载选中图片")
            download_output_file = gr.File(label="下载文件", interactive=False) # 用于显示下载链接
    # --- 事件处理 ---
    text_search_button.click(
        handle_text_search,
        inputs=[text_query_input, favorites_list_state],
        outputs=[results_gallery, search_status_message, current_gallery_images]
    )
    image_search_button.click(
        handle_image_search,
        inputs=[image_query_input, favorites_list_state],
        outputs=[results_gallery, search_status_message, current_gallery_images]
    )

    # Refinement: 当用户在Gallery中选择一张图片时，使用它进行新的搜索
    # 同时，更新选中图片路径，并显示收藏/下载按钮
    def gallery_select_handler(evt: gr.SelectData, fav_list_state):
        if evt.value: # evt.value 是选中图片的路径
            # Action 1: Refinement by re-searching with this image
            new_gallery, new_msg, updated_fav_list_state = search_by_selected_image(evt,fav_list_state)
            # Action 2: Prepare for favorite/download
            # 使操作按钮可见，并设置当前选中的图片路径
            return {
                results_gallery: new_gallery,
                search_status_message: new_msg if new_gallery else f"已选择图片'{os.path.basename(evt.value)}' 进行操作。原搜索：{new_msg}",
                action_buttons_row: gr.Row(visible=True),
                selected_image_path_for_actions: evt.value,
                current_gallery_images: updated_fav_list_state # Or new_gallery if you want to update current_gallery_images
            }
        # 如果没有选择图片（例如取消选择），则隐藏操作按钮
        return {
            action_buttons_row: gr.Row(visible=False),
            selected_image_path_for_actions: None,
        }

    # Gradio 的 Gallery select 事件现在仅用于选择图片进行操作，而不是直接再次搜索
    # 我们将 "点击图片再搜索" 的逻辑放在一个单独的按钮或者明确的指示上
    # 为简化，我们将 select 事件用于更新 "selected_image_path_for_actions" 和显示按钮
    def gallery_select_for_action(evt: gr.SelectData):
        if evt.value:
            return {
                action_buttons_row: gr.Row(visible=True),
                selected_image_path_for_actions: evt.value,
                # search_status_message: f"已选中图片 '{os.path.basename(evt.value)}' 进行收藏或下载。"
            }
        return {
            action_buttons_row: gr.Row(visible=False),
            selected_image_path_for_actions: None
        }
    
    # results_gallery.select(gallery_select_handler, inputs=[favorites_list_state], outputs=[results_gallery, search_status_message, action_buttons_row, selected_image_path_for_actions, current_gallery_images])
    results_gallery.select(
        gallery_select_for_action,
        outputs=[action_buttons_row, selected_image_path_for_actions] # removed search_status_message
    )

    # Refinement: 点击结果中的图片进行再次搜索 (通过一个显式按钮或文字提示)
    # 这里我们依赖用户理解 "点击图片=再次搜索" 的说明，并通过 gallery.select触发。
    # 如果想更明确，可以为每个图片添加一个小的"以此图搜索"按钮，但这会使Gallery复杂化。
    # 上面的 gallery_select_handler 已经实现了点击图片再搜索的逻辑，但为了分离关注点，可以调整。
    # 当前: 点击图片会触发 gallery_select_for_action，它仅准备后续操作。

    # 若要实现点击图片直接再搜索，需要调整 select 的回调。
    # 为了满足题目要求的 "点击得到按相似度排序的结果"，我们修改 select 逻辑。
    # (选择1: 点击图片直接再搜索 - 如上面的 gallery_select_handler)
    # (选择2: 点击图片准备操作，用户再点其他按钮执行操作 - 当前的 gallery_select_for_action)
    # 为了满足"以图像搜索图像，并得到按相似度排序的结果 (2分)"且通过点击结果图片，
    # 我们让 gallery.select 直接触发以图搜图。
    # 同时，我们需要一种方式来收藏/下载图片。
    # 这造成了交互冲突：点击是再搜索还是选中操作？
    # 解决方案：
    # 1. Gallery 点击 -> 以图搜图 (Refinement)
    # 2. 在Gallery下方提供一个 "对最后点击的图片进行操作" 的区域，或者
    # 3. 改变Gallery的交互，例如长按或右键出菜单（Gradio标准组件不支持）
    # 4. 在每个图片下方显示小的操作按钮（需要自定义HTML或更高级的Gradio用法）
    # 我们选择方案1，并为收藏/下载提供一个略微不同的流程：
    # 用户搜索 -> 看到结果 -> 点击一张图片 (触发以此图搜索) ->
    # 如果用户想对 *某个搜索结果* 进行收藏/下载，他们需要另一种方式指定。
    # 一个简单的方式是，当图片展示在Gallery时，就允许收藏/下载，而不是仅针对"选中"的图片。
    # 但题目要求 "其他任意能够提升用户交互体验的功能（如收藏、下载等，1分）"
    # 让我们保留 "selected_image_path_for_actions" 并让用户明确点击它来收藏/下载。
    # 这样，`results_gallery.select` 主要用于选中图片以进行 *后续操作* (收藏/下载)
    # 而 "Refinement" (以结果图片搜索) 需要一个更明确的触发，比如在图片旁边加个小按钮 (Gradio不易)
    # 或提供一个 "用选中的图片搜索" 的按钮。
    # 为了满足要求"用户能够使用图像进行图像搜索，并得到按相似度排序的结果 (2分)"
    # 其中一种方式是点击Gallery中的图片。让我们恢复这个逻辑。
    def gallery_click_refines_and_selects(evt: gr.SelectData, fav_list_state):
        """当点击gallery中的图片时:
        1. 以该图片作为查询进行新的搜索 (Refinement)。
        2. 同时将该图片设置为当前选中，以便进行收藏/下载操作。
        """
        if not evt.value: # 如果没有图片被选中（例如，取消选择）
            return {
                results_gallery: [], # 清空画廊或保持不变
                search_status_message: "未选择图片进行操作。",
                action_buttons_row: gr.Row(visible=False),
                selected_image_path_for_actions: None,
                current_gallery_images: fav_list_state # 或者保持 current_gallery_images 不变
            }
        selected_image_path = evt.value
        print(f"Refinement: Clicked on result image '{os.path.basename(selected_image_path)}' to start new search.") # Stage 4
        if selected_image_path and os.path.exists(selected_image_path):
            try:
                pil_image = Image.open(selected_image_path)
                # 执行以图搜图
                new_gallery_paths, message, updated_fav_list_state = handle_image_search(pil_image, fav_list_state)
                return {
                    results_gallery: new_gallery_paths,
                    search_status_message: message,
                    action_buttons_row: gr.Row(visible=True), # 显示操作按钮
                    selected_image_path_for_actions: selected_image_path, # 设置当前选中图片以备操作
                    current_gallery_images: new_gallery_paths # 更新当前画廊图片
                }
            except Exception as e:
                error_msg = f"无法加载或搜索选中图片: {str(e)}"
                return {
                    results_gallery: [],
                    search_status_message: error_msg,
                    action_buttons_row: gr.Row(visible=False),
                    selected_image_path_for_actions: None,
                    current_gallery_images: fav_list_state
                }
        else:
            no_path_msg = "选中的图片路径无效。"
            return {
                results_gallery: [],
                search_status_message: no_path_msg,
                action_buttons_row: gr.Row(visible=False),
                selected_image_path_for_actions: None,
                current_gallery_images: fav_list_state
            }
    results_gallery.select(
        gallery_click_refines_and_selects,
        inputs=[favorites_list_state],
        outputs=[results_gallery, search_status_message, action_buttons_row,
        selected_image_path_for_actions, current_gallery_images]
    )

    # 收藏和下载按钮的事件
    add_to_favorites_button.click(
        add_to_favorites,
        inputs=[selected_image_path_for_actions, favorites_list_state],
        outputs=[favorites_list_state, favorites_status_message, current_gallery_images]
        # current_gallery_images is actually favorites_list_state
    ).then(
        lambda fav_list: fav_list, # 这个 then 只是为了在点击后能更新 current_gallery_images (用作favorites_list_state 的副本)
        inputs=[favorites_list_state],
        outputs=[current_gallery_images] # 实际上这里 current_gallery_images 会被favorites_list_state 更新
    )

    download_selected_button.click(
        download_image_action,
        inputs=[selected_image_path_for_actions],
        outputs=[download_output_file, search_status_message] # search_status_message is used for download status
    )

    view_favorites_button.click(
        view_favorites,
        inputs=[favorites_list_state],
        outputs=[results_gallery, search_status_message, current_gallery_images] # Display favorites in main gallery
    )
    clear_favorites_button.click(
        clear_favorites,
        inputs=[favorites_list_state],
        outputs=[favorites_list_state, favorites_status_message, current_gallery_images]
    ).then(
        lambda: [], # Clear the main gallery when favorites are cleared
        outputs=[results_gallery]
    )



if __name__ == "__main__":
    demo.launch(debug=True)