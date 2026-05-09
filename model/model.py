"""
结合 ResNet 的残差连接与 MobileNet 的深度可分离卷积，

数据流概览：
输入: (batch_size, 3, H, W) 经过标准化的 RGB 图像
  ↓ stem 卷积
特征图: (batch_size, 32, H/2, W/2)
  ↓ 多个带残差的深度可分离卷积块 (逐步下采样/增加通道)
特征图: (batch_size, 1280, H/32, W/32)
  ↓ 全局平均池化
特征向量: (batch_size, 1280)
  ↓ 全连接分类器
输出: (batch_size, num_classes) - 各类别的 logits
"""

import torch
import torch.nn as nn


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    """
    标准卷积 + BatchNorm + ReLU6 组合
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(in_channels, out_channels):
    """1x1 卷积 + BatchNorm + ReLU6"""
    return conv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = 逐通道卷积 (depthwise) + 逐点卷积 (pointwise)
    参数量和计算量远小于标准卷积。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # depthwise: 每个通道单独做卷积，groups=in_channels
        self.depthwise = conv_bn(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        # pointwise: 1x1 卷积混合通道
        self.pointwise = conv_1x1_bn(in_channels, out_channels)

    def forward(self, x):
        # 输入: (N, C_in, H, W)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """
    倒残差模块 (源自 MobileNetV2)
    操作流程:
        输入 -> 1x1 扩展 (expand) -> 深度可分离卷积 -> 1x1 压缩 (project) -> 与输入相加 (若 shape 相同)
    特点：先升维，再降维，中间使用深度可分离卷积。
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # 扩张层 (当 expand_ratio != 1)
        if expand_ratio != 1:
            layers.append(conv_1x1_bn(in_channels, hidden_dim))
        # 深度可分离卷积
        layers.append(DepthwiseSeparableConv(hidden_dim, out_channels, stride=stride))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # 输入: (N, C_in, H, W)
        if self.use_residual:
            return x + self.conv(x)  # 残差连接
        else:
            return self.conv(x)


class MiniMobileResNet(nn.Module):
    """
    轻量级垃圾分类模型
    参数总量约 1.5 M，适合树莓派 5。
    输入图像尺寸建议 224x224 (自适应也可)。
    """
    def __init__(self, num_classes=55, input_channels=3):
        super().__init__()

        # ---- 第一阶段: stem 卷积 (快速降低分辨率) ----
        # 输入: (N, 3, 224, 224) -> 输出: (N, 32, 112, 112)
        self.stem = nn.Sequential(
            conv_bn(input_channels, 32, stride=2),   # 下采样 2x
        )

        # ---- 主体: 多个倒残差块组成的阶段 ----
        # 各阶段配置: [in, out, stride, expand_ratio, 重复次数]
        # 注: 仅每个阶段的第一层负责改变通道/分辨率
        stage_configs = [
            # t, c, n, s (expand_ratio, out_channels, num_blocks, stride)
            [1,  16, 1, 1],   # 第一阶段: 输出 16 通道，不改变分辨率
            [6,  24, 2, 2],   # 第二阶段: 输出 24 通道，分辨率减半 (56x56)
            [6,  32, 3, 2],   # 第三阶段: 输出 32 通道，分辨率减半 (28x28)
            [6,  64, 4, 2],   # 第四阶段: 输出 64 通道，分辨率减半 (14x14)
            [6,  96, 3, 1],   # 第五阶段: 输出 96 通道，保持分辨率
            [6, 160, 3, 2],   # 第六阶段: 输出 160 通道，分辨率减半 (7x7)
            [6, 320, 1, 1],   # 第七阶段: 输出 320 通道，保持分辨率
        ]

        # 初始输入通道数为 stem 的输出 32
        in_ch = 32
        self.features = []
        for t, c, n, s in stage_configs:
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(in_ch, c, stride, expand_ratio=t))
                in_ch = c  # 后续块的输入通道更新为当前输出
        self.features = nn.Sequential(*self.features)

        # ---- 最后卷积层: 将通道数升至 1280 ----
        # 输入: (N, 320, 7, 7) -> 输出: (N, 1280, 7, 7)
        self.last_conv = conv_1x1_bn(in_ch, 1280)

        # ---- 分类器 ----
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，输出 (N, 1280, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播数据流:
        x: tensor, 形状 (batch_size, 3, H, W) - 已归一化的图像
        """
        # stem: (N, 3, 224, 224) -> (N, 32, 112, 112)
        x = self.stem(x)

        # 倒残差块: 逐步提取特征并下采样
        # -> (N, 16, 112, 112) -> (N, 24, 56, 56) -> (N, 32, 28, 28)
        # -> (N, 64, 14, 14) -> (N, 96, 14, 14) -> (N, 160, 7, 7) -> (N, 320, 7, 7)
        x = self.features(x)

        # 升维: (N, 320, 7, 7) -> (N, 1280, 7, 7)
        x = self.last_conv(x)

        # 全局平均池化: (N, 1280, 7, 7) -> (N, 1280, 1, 1)
        x = self.avg_pool(x)
        # 展平: (N, 1280)
        x = x.view(x.size(0), -1)

        # 分类器: (N, 1280) -> (N, num_classes)  -> 每个类别的 logits
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ------------------- 简单测试 -------------------
if __name__ == '__main__':
    model = MiniMobileResNet(num_classes=55)
    # 模拟一批图像: Batch=2, 3 通道, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # 预期: (2, 55)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")