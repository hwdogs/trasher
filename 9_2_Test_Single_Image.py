"""
单图垃圾分类测试脚本（最终版）
=============================
功能：
  - 自动生成 class_names.txt（若缺失）
  - 通过命令行指定图片、模型、数据集路径
  - 输出预测类别、置信度、推理时间
  - 显示 Top-5 预测结果

用法：
  python test.py --img path/to/image.jpg [--data_root path/to/dataset] [--model_path path/to/model.pth]

数据流：
  图片 (PIL) → padding + transform → Tensor (1,3,224,224)
          → MiniMobileResNet → logits (1,num_classes)
          → softmax → probs → 取 top-5 → 输出类别+置信度
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model.model import MiniMobileResNet   # 确保路径正确

# ==================== 配置 ====================
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 默认路径（可通过命令行覆盖）
DEFAULT_DATA_ROOT = r"..\dataset\dataset_55_4_enhance"
DEFAULT_MODEL_PATH = r"model_pth\minimobileresnet_best.pth"
CLASS_NAMES_FILE = "class_names.txt"

# 标准化参数（与训练一致）
MEAN = [0.645832, 0.579004, 0.513855]
STD  = [0.294306, 0.299911, 0.329036]

# ==================== 图像预处理 ====================
def padding_black(img, img_size=IMG_SIZE):
    """保持宽高比缩放，并填充黑色边至正方形"""
    w, h = img.size
    scale = img_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    bg = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    bg.paste(img, ((img_size - new_w) // 2, (img_size - new_h) // 2))
    return bg

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ==================== 类别名称处理 ====================
def generate_class_names(data_root, output_file):
    """遍历数据集子文件夹，按名称排序后写入文件"""
    subdirs = [d for d in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, d))]
    if not subdirs:
        raise RuntimeError(f"在 {data_root} 下未找到任何类别文件夹")
    subdirs.sort()
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in subdirs:
            f.write(name + '\n')
    print(f"已生成类别名称文件：{output_file}，共 {len(subdirs)} 个类别")
    return subdirs

def load_class_names(file_path, data_root):
    """加载类别名称，若文件不存在则自动生成"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    else:
        print(f"未找到 {file_path}，从数据集自动生成...")
        return generate_class_names(data_root, file_path)

# ==================== 模型加载 ====================
def load_model(model_path, num_classes):
    model = MiniMobileResNet(num_classes=num_classes).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="垃圾分类单图测试")
    parser.add_argument('--img', required=True, help='待测试的图片路径')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT,
                        help='数据集根目录（用于生成 class_names.txt）')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH,
                        help='训练好的 .pth 模型路径')
    args = parser.parse_args()

    # 1. 加载类别名称
    class_names = load_class_names(CLASS_NAMES_FILE, args.data_root)
    num_classes = len(class_names)
    print(f"类别数量：{num_classes}")

    # 2. 加载模型
    model = load_model(args.model_path, num_classes)
    print(f"模型已加载到 {DEVICE}")

    # 3. 预处理图片
    img = Image.open(args.img).convert('RGB')
    img = padding_black(img)
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    # 4. 推理
    print("正在推理...")
    with torch.no_grad():
        t0 = time.time()
        logits = model(img_tensor)                     # (1, num_classes)
        t1 = time.time()

        probs = F.softmax(logits, dim=1)               # 概率分布
        top5_prob, top5_idx = torch.topk(probs, k=5, dim=1)  # top5 索引和概率

    # 5. 输出
    print("\n" + "=" * 50)
    print("          垃圾分类预测结果")
    print("=" * 50)
    for rank in range(5):
        idx = top5_idx[0, rank].item()
        prob = top5_prob[0, rank].item()
        print(f"Top-{rank+1}: {class_names[idx]:<25s} 置信度: {prob:.4f} ({prob*100:.2f}%)")
    print(f"\n推理耗时: {(t1 - t0) * 1000:.2f} ms")
    print("=" * 50)

if __name__ == '__main__':
    main()