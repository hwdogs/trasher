from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms as T
from tqdm import tqdm

transform = T.Compose([
    T.Resize(256),          # 计算统计量时使用固定的转换
    T.CenterCrop(224),
    T.ToTensor(),
])
 
def getStat(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
 
    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for X, _ in tqdm(train_loader):
    #     for d in range(3):
    #         mean[d] += X[:, d, :, :].mean() # N, C, H ,W
    #         std[d] += X[:, d, :, :].std()
    # mean.div_(len(train_data))
    # std.div_(len(train_data))
    # return list(mean.numpy()), list(std.numpy())

    total_sum = torch.zeros(3)
    total_sq = torch.zeros(3)
    total_pixels = 0

    for X, _ in tqdm(train_loader):
        batch_pixels = X.shape[0] * X.shape[2] * X.shape[3]
        total_pixels += batch_pixels
        total_sum += X.sum(dim=(0, 2, 3))
        total_sq += (X ** 2).sum(dim=(0, 2, 3))

    mean = total_sum / total_pixels
    std = torch.sqrt(total_sq / total_pixels - mean ** 2)
    return mean.tolist(), std.tolist()
 
 
if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'..\dataset\dataset_55_4_enhance', transform=transform)
    mean, std = getStat(train_dataset)
    print(f"mean = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"std = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")

    # 覆盖写入文件（保留6位，格式干净，可直接复制进代码）
    with open('dataset_mean_std.txt', 'w') as f:
        f.write(f"mean = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]\n")
        f.write(f"std = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]\n")