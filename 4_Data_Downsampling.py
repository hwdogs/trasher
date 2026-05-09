import os
import random
import importlib

Data_Analysis = importlib.import_module('1_Data_Analysis')

dataset_root_path = r'..\dataset\dataset_55_4_enhance'
threshold = 300

# 遍历所有子目录，对每个包含文件的目录进行处理
for root, dirs, files in os.walk(dataset_root_path):
    if not files:
        continue   # 跳过空目录

    total_files = len(files)

    if total_files > threshold:
        # 随机选择需要保留的文件（保留 threshold 个）
        keep_files = random.sample(files, threshold)
        # 需要删除的文件 = 全集 - 保留集
        delete_files = [f for f in files if f not in keep_files]

        deleted_count = 0
        for file_name in delete_files:
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"删除失败: {file_path}，错误: {e}")

        print(f"目录: {root}")
        print(f"  原始文件数: {total_files}, 目标保留数: {threshold}, 实际删除数: {deleted_count}\n")
    else:
        print(f"目录: {root} 文件数 {total_files} ≤ 阈值 {threshold}，无需处理\n")

Data_Analysis.plot_resolution(dataset_root_path, operation='下采样')