'''
    使用翻转进行数据增强 对于图片样本比较少的类别
    mean = 208.98 -》300+-
'''
import os
import cv2
import numpy as np
import importlib

Data_Analysis = importlib.import_module('1_Data_Analysis')

# 水平翻转
def Horizontal(image):
    return cv2.flip(image, 1, dst=None)  # 水平镜像

# 垂直翻转
def Vertical(image):
    return cv2.flip(image, 0, dst=None)  # 垂直镜像

if __name__ == '__main__':
    from_root = r'..\dataset\dataset_55_4'
    save_root = r'..\dataset\dataset_55_4_enhance'
    threshold = 200

    for root, dirs, files in os.walk(from_root):
        # 当前目录下的文件数量（即该类别样本数）
        num_files = len(files)

        # 判断当前类别是否需要增强（样本数少于阈值）
        need_enhance = num_files < threshold

        for file_i in files:
            file_i_path = os.path.join(root, file_i)

            # 提取当前文件所属的类别名（最后一层文件夹名）
            dir_loc = os.path.basename(root)
            save_path = os.path.join(save_root, dir_loc)

            # 创建目标目录（如果不存在）
            os.makedirs(save_path, exist_ok=True)

            # 读取图片（支持中文路径）
            img_i = cv2.imdecode(np.fromfile(file_i_path, dtype=np.uint8), -1)

            # 始终保存原始图片（重命名为 *_original.jpg）
            original_save_name = file_i[:-5] + "_original.jpg"
            cv2.imencode('.jpg', img_i)[1].tofile(os.path.join(save_path, original_save_name))

            # 如果当前类别样本数低于阈值，则生成水平/垂直翻转图片
            if need_enhance:
                img_horizontal = Horizontal(img_i)
                hori_save_name = file_i[:-5] + "_horizontal.jpg"
                cv2.imencode('.jpg', img_horizontal)[1].tofile(os.path.join(save_path, hori_save_name))

                img_vertical = Vertical(img_i)
                vert_save_name = file_i[:-5] + "_vertical.jpg"
                cv2.imencode('.jpg', img_vertical)[1].tofile(os.path.join(save_path, vert_save_name))

    # 调用分析模块，生成增强后的分辨率统计图
    Data_Analysis.plot_resolution(save_root, operation='增强')