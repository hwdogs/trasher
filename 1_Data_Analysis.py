'''
    数据样本分析
    功能：画出数据量条形图 + 画出图像分辨率散点图
'''
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
 
def plot_resolution(dataset_root_path, operation = 'nomal'):
    """
    综合数据分析函数
    同时统计：
    1. 图片尺寸信息（用于散点图）
    2. 各类别文件数量（用于条形图）
    """
    # 初始化数据容器
    img_size_list = []      # 存储所有图片的(宽, 高)元组
    file_name_list = []     # 存储所有类别名称
    file_num_list = []      # 存储每个类别的图片数量
    for root, dirs, files in os.walk(dataset_root_path):
        # root = ..\dataset\dataset_55_4
        # files = ['.\厨余垃圾_巴旦木\img_巴旦木_1.jpeg', ...]

        # 统计图片尺寸信息
        for file_i in files:
            file_i_full_path = os.path.join(root, file_i)   # 文件完整路径 ..\dataset\dataset_55_4\厨余垃圾_巴旦木\img_巴旦木_1.jpeg"
            img_i = Image.open(file_i_full_path)            # 用PIL打开图片
            img_i_size = img_i.size                         # 获取单张图像的宽高(宽, 高)
            img_size_list.append(img_i_size)                # 保存图片尺寸信息[(256，256), ...]

        # 统计得到每一类文件总数
        if len(dirs) != 0:                                  # 确保是在最外面的根目录进行操作
            for dir_i in dirs:                              # 读取根目录下所有文件夹名称
                file_name_list.append(dir_i)
        file_num_list.append(len(files))                    # 得到每一类图片总数


    # 统计图片尺寸信息
    print(img_size_list[:50], '\n图片总数：', len(img_size_list))
    width_list = [img_size_list[i][0] for i in range(len(img_size_list))]   # 读取列表中每个数据的第一维
    height_list = [img_size_list[i][1] for i in range(len(img_size_list))]  # 读取列表中每个数据的第二维


    # 统计得到每一类文件总数
    file_num_list = file_num_list[1:]                       # 去掉根目录下对文件夹的统计
    # 求均值，并把均值以横线形式显示出来
    mean = np.mean(file_num_list)
    print("mean = ", mean)

    plt.rcParams["font.sans-serif"] = ["SimHei"]    # 设置中文字体
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.unicode_minus"] = False      # 该语句解决图像中的“-”负号的乱码问题
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    bar_positions = np.arange(len(file_name_list))
    fig, ax = plt.subplots()                                            # 定义画的区间和子画
    ax.bar(bar_positions, file_num_list, 0.5)                           # 画柱图，参数：柱间的距离，柱的值，柱的宽度
    ax.plot(bar_positions, [mean for i in bar_positions], color="red")  # 显示平均值
    ax.set_xticks(bar_positions)  # 设置x轴的刻度
    ax.set_xticklabels(file_name_list, rotation=90)  # 设置x轴的标签
    ax.set_ylabel("类别数量")
    ax.set_title("数据分布图")
    plt.savefig(fr'out\每个类别的图像数量_{operation}_{timestamp}.png')
    plt.show()
 
 
    plt.scatter(width_list, height_list, s=1)       # 散点图
    plt.xlabel("宽")
    plt.ylabel("高")
    plt.title("图像宽高分布")
    plt.savefig(fr'out\图像宽高分布_{operation}_{timestamp}.png')
    plt.show()                                      # 问题1：样本分辨率分布不均匀
 
if __name__ == '__main__':
 
    # dataset_root_path = r"\dataset\dataset_55_4"
 
    # plot_resolution(dataset_root_path)
 
    # plot_bar(dataset_root_path)
    try:
        print('所有库已导入成功')
        dataset_root_path = r"..\dataset\dataset_55_4"
        plot_resolution(dataset_root_path)
    except Exception as e:
        import traceback
        print("="*60)
        print("程序发生致命错误，被外层 try...except 捕获：")
        traceback.print_exc()  # 这会打印出完整的错误堆栈
        print("="*60)
    finally:
        # 无论是否出错，都暂停一下
        input("按回车键退出...")