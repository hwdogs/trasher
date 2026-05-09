'''
    图片处理
    图像宽高保持在200到2000
    排除图像宽高比低于0.5的数据
    不符合要求的图片会被移动到 dataset_delete 目录，并保留原有子文件夹结构
'''

from PIL import Image
import os
import shutil
import importlib

Data_Analysis = importlib.import_module('1_Data_Analysis')
dataset_root_path = r"..\dataset\dataset_55_4"
dataset_delete = r"..\dataset\dataset_55_4_delete"

min_size = 200   # 短边最小值
max_size = 2000  # 长边最大值
ratio = 0.5      # 短边 / 长边 最低比例

delete_list = []  # 用于存放所有不符合条件的文件路径

for root, dirs, files in os.walk(dataset_root_path):
    for file_i in files:
        file_i_full_path = os.path.join(root, file_i)
        img_i = Image.open(file_i_full_path)
        img_i_size = img_i.size  # (width, height)

        need_delete = False

        # 条件1：单边过短
        if img_i_size[0] < min_size or img_i_size[1] < min_size:
            print(file_i_full_path, "不满足要求（单边过短）")
            need_delete = True

        # 条件2：单边过长
        if img_i_size[0] > max_size or img_i_size[1] > max_size:
            print(file_i_full_path, "不满足要求（单边过长）")
            need_delete = True

        # 条件3：宽高比例不当
        long_side = max(img_i_size[0], img_i_size[1])
        short_side = min(img_i_size[0], img_i_size[1])
        if short_side / long_side < ratio:
            print(file_i_full_path, "不满足要求（宽高比过低）", img_i_size[0], img_i_size[1])
            need_delete = True

        if need_delete and file_i_full_path not in delete_list:
            delete_list.append(file_i_full_path)

# 移动不符合条件的文件到 dataset_delete，保留子文件夹结构
for file_path in delete_list:
    # 计算相对路径（相对于 dataset_root_path）
    relative_path = os.path.relpath(file_path, dataset_root_path)
    # 目标完整路径
    dest_path = os.path.join(dataset_delete, relative_path)
    dest_dir = os.path.dirname(dest_path)

    try:
        # 创建目标子目录（如果不存在）
        os.makedirs(dest_dir, exist_ok=True)
        # 移动文件
        shutil.move(file_path, dest_path)
        print("已移动:", file_path, "->", dest_path)
    except Exception as e:
        print("移动失败:", file_path, "错误:", e)

print('已移动图片数量：', len(delete_list))

Data_Analysis.plot_resolution(dataset_root_path, operation='剔除')