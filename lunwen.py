"""
数据集概览脚本
功能：统计图片总数与总大小，随机展示4张样本图片
用法：python dataset_overview.py --data_root path/to/dataset
"""
import os
import sys
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

def format_size(size_bytes):
    """将字节数转换为可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    # 统一字体设置（加大基础字号）
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.size"] = 12          # 全局字体大小统一为12
    plt.rcParams["axes.unicode_minus"] = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="数据集概览 - 统计信息与随机样本展示")
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录路径')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"错误：路径 {data_root} 不存在")
        sys.exit(1)

    # 收集所有图片文件路径（不区分扩展名大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    all_images = []
    total_size = 0

    # 一次性遍历所有文件，根据扩展名（小写）判断是否图片
    for file_path in data_root.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            all_images.append(file_path)
            total_size += file_path.stat().st_size

    total_images = len(all_images)

    if total_images == 0:
        print("错误：在数据集目录中未找到任何图片文件")
        sys.exit(1)

    print(f"数据集路径: {data_root.absolute()}")
    print(f"图片总数:   {total_images} 张")
    print(f"总数据大小: {format_size(total_size)}")

    # 统计各类别文件夹及图片数量
    category_counts = {}
    for img_path in all_images:
        category = img_path.parent.name
        category_counts[category] = category_counts.get(category, 0) + 1
    print(f"类别数量:   {len(category_counts)} 个")
    print(f"各类别图片数量范围: {min(category_counts.values())} ~ {max(category_counts.values())} 张")

    # 随机选取4张图片展示
    samples = random.sample(all_images, min(4, total_images))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    # 总标题字体可以稍大，但保持风格统一（例如使用全局字体+2）
    fig.suptitle(f'数据集随机样本概览\n图片总数: {total_images} 张 | 总大小: {format_size(total_size)}',
                 fontsize=plt.rcParams["font.size"] + 2)

    for i, (img_path, ax) in enumerate(zip(samples, axes)):
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)

            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'加载失败\n{str(e)}', ha='center', va='center')
            ax.axis('off')

    for i in range(len(samples), 4):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()