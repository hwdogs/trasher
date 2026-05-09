import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（避免乱码）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者 ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示问题

def parse_log_file(log_path):
    """
    解析训练日志文件
    实际格式：epoch\t0\ttrain_loss\t3.6755\tval_loss\t3.3959\tval_accuracy\t0.0896
    提取 epoch, train_loss, val_loss, val_accuracy
    """
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            # 期望每行至少有8个字段（标签+数值交替）
            if len(parts) < 8:
                continue
            try:
                # 提取第二个字段（epoch值）、第四个（train_loss）、第六个（val_loss）、第八个（val_accuracy）
                epoch = int(parts[1])
                train_loss = float(parts[3])
                val_loss = float(parts[5])
                val_acc = float(parts[7])
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            except (ValueError, IndexError):
                # 跳过无法解析的行（例如标题行或格式错误）
                continue
    return epochs, train_losses, val_losses, val_accs

def plot_curves(epochs, train_losses, val_losses, val_accs, save_path=None):
    """
    绘制两条曲线：损失曲线（train & val）和准确率曲线（val）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('训练与验证损失曲线', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右图：验证准确率曲线
    ax2.plot(epochs, val_accs, 'g-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('验证准确率曲线', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    # 请根据实际情况修改日志文件的路径
    log_file = "model_pth/minimobileresnet.txt"   # 你实际保存的日志文件名
    if not os.path.exists(log_file):
        log_file = "model_pth/resnet18_no_pretrain.txt"

    if not os.path.exists(log_file):
        print(f"错误：找不到日志文件 {log_file}，请检查路径。")
    else:
        epochs, train_losses, val_losses, val_accs = parse_log_file(log_file)
        if len(epochs) == 0:
            print("日志文件中没有有效数据，请检查文件格式。")
        else:
            # 绘制并保存图片
            plot_curves(epochs, train_losses, val_losses, val_accs,
                        save_path="training_curves.png")
            print("曲线绘制完成！")