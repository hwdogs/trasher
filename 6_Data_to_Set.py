import os
import random
from tqdm import tqdm
 
train_ratio = 0.9
test_ratio = 1-train_ratio
 
dataset_root_path = r'..\dataset\dataset_55_4_enhance'
output_train = 'train.txt'
output_test = 'test.txt'
 
train_list, test_list = [],[]
class_label = 0

for root, dirs, files in os.walk(dataset_root_path):
    if not files:
        continue    # 跳过空文件目录

    random.shuffle(files)   # 对当前类别的文件列表随机打乱，保证分割随机性
    train_cnt = int(len(files)* train_ratio)

    train_files = files[:train_cnt]     # 分割训练/测试
    test_files = files[train_cnt:]

    for f in train_files:
        train_data_path = os.path.join(root, f)
        train_list.append(f'{train_data_path}\t{class_label}\n')
 
    for f in test_files:
        test_data_path = os.path.join(root, f)
        test_list.append(f'{test_data_path}\t{class_label}\n')
 
    class_label += 1
 
random.shuffle(train_list)
random.shuffle(test_list)
 
with open(output_train,'w',encoding='UTF-8') as f:
    f.writelines(train_list)
 
with open(output_test,'w',encoding='UTF-8') as f:
    f.writelines(test_list)