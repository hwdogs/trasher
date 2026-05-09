import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
import importlib

from model.model import MiniMobileResNet

LoadDataUtils = importlib.import_module('7_Load_Data_Utils')
 
def train(dataloader, model, loss_fn, optimizer,device):
    model.train()

    size = len(dataloader.dataset)
    total_loss = 0.0
    
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    pbar = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for batch, (X, y) in enumerate(pbar):#固定格式：batch：第几批数据，不是批次大小，（X，y）：数值用括号
        # 将数据存到显卡
        X, y = X.to(device), y.to(device)
        # 得到预测的结果pred
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * X.size(0)   # 总损失
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每训练10次，输出一次当前信息
        if batch % 10 == 0:
            # loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            pbar.set_postfix(loss=loss.item(), current=batch * len(X))
 
    # 当一个epoch完了后返回平均 loss
    avg_loss = total_loss / size   # 每个样本的平均损失
    return avg_loss
 
 
def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)                               # 将数据转到GPU
            pred = model(X)                                                 # 将图片传入到模型当中就，得到预测的值pred
            loss =  loss_fn(pred, y)                                        # 计算预测值pred和真实值y的差距
            test_loss += loss.item() * X.size(0)                            # 恢复为该 batch 的总损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 统计预测正确的个数(针对分类)
    test_loss /= size
    correct /= size
    print(f"correct = {correct}, Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss
 
 
if __name__=='__main__':
    torch.cuda.empty_cache()    # 清空 CUDA 缓存
    torch.cuda.reset_peak_memory_stats()
    batch_size = 16
    # batch_size = 64
 
    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadDataUtils.LoadData("train.txt", True)
    valid_data = LoadDataUtils.LoadData("test.txt", False)
 
 
    train_dataloader = DataLoader(dataset=train_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size, shuffle=False)
 
    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
 
 
    # model = resnet18(num_classes=55).to(device)
    model = MiniMobileResNet(num_classes=55).to(device)
 
    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()
 
    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    learning_rate = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.AdamW(
        model.parameters(), # 提供模型需要优化的参数
        lr=1e-3,            # 学习率
        betas=(0.9, 0.999), # 一阶与二阶动量的衰减系数
        eps=1e-8,           # 防止除零的小常数
        weight_decay=1e-2   # L2 正则化
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',     # 默认值，当Loss不再下降时降速
        factor=0.5, 
        patience=5
    )
 
    # epochs = 10
    epochs = 50
    best_val_loss = float('inf')
    save_root = "model_pth/"
    os.makedirs(save_root, exist_ok=True)
 
 
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        time_start = time.time()
        avg_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        time_end = time.time()
        print(f"train time: {(time_end - time_start)}")
        # (dataloader, model, loss_fn, device)jif
        val_accuracy, val_loss = validate(valid_dataloader, model,loss_fn, device)
        # 写入数据
        LoadDataUtils.WriteData(save_root + "minimobileresnet.txt",
                  "epoch", t,
                  "train_loss", avg_loss,
                  "val_loss", val_loss,
                  "val_accuracy", val_accuracy)
        if t % 5 == 0:
            torch.save(model.state_dict(), f"{save_root}minimobileresnet_epoch{t}_loss_{avg_loss:.4f}.pth")
 
        torch.save(model.state_dict(), f"{save_root}minimobileresnet_last.pth")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_root}minimobileresnet_best.pth")
        
        # scheduler.step()
        scheduler.step(val_loss)    # ReduceLROnPlateau基于验证损失自动降低学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")