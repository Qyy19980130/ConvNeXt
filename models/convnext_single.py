"""
imm 库中的 trunc_normal_ 和 DropPath 用于初始化权重和实现深度随机丢弃。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchsummary import summary

'''
Block 类是 ConvNeXt 中的一个基本组件，它实现了一个由多层操作构成的残差块。主要包括:
    dwconv：一个深度可分离卷积 (Depthwise Convolution)，仅在通道维度上进行卷积操作，保持输入的通道数不变。
    LayerNorm：用于标准化输入数据。可以对不同数据格式（如 channels_last 或 channels_first）进行标准化。也就是论文提到的LN
    pwconv1 和 pwconv2：实现点卷积（1x1 卷积），分别扩大和缩小通道数。
    激活函数 (GELU)：使用 GELU 激活函数来增加非线性。
    DropPath：实现随机深度（Stochastic Depth），这是对残差路径随机丢弃的一种正则化技术。
forward 方法：完成输入经过卷积、标准化、激活和残差连接的过程。

'''


class Block(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): 输入的通道数。
        drop_path (float):  随机丢弃率，用于 DropPath 操作。
        layer_scale_init_value (float): 初始层缩放值，用于调整输出的幅度。
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None  # 用于调整输出比例。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath 层，用于随机丢弃路径。

    """
        逐步实现了深度卷积、维度变换、归一化、通道映射、激活、残差连接、以及 DropPath。主要是深度卷积和线性映射，最后加入了残差连接。
    """

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)  # 残差链接
        return x


class ConvNeXt(nn.Module):
    # depths和dims 表示每个阶段的残差块数量和特征通道数量  drop_path_rate：随机深度的丢弃率。
    def __init__(self, in_chans, num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers  存储下采样层的模块列表。
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=(1, 3), stride=(4, 2), padding=(3, 1)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )  # 第一个下采样层，包含一个卷积和 LayerNorm 操作。
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=(1, 1), stride=(2, 1)),
            )  # 继续添加三层下采样层，每一层都通过卷积进行通道数扩展和空间下采样。
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks  存储每个阶段的模块列表。
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 计算每个 Block 的 drop_path 参数。
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # 逐步经过下采样层和残差块，最后进行归一化和全局平均池化。
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    # 最后调用 forward_features 处理特征，并通过线性分类头输出结果。
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


'''
自定义的 LayerNorm 支持两种数据格式。
normalized_shape: 要归一化的通道数。
eps: 防止除零的极小值。
data_format: 数据格式，可以是 "channels_last" 或 "channels_first"。
'''


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    # Step 1: 实例化模型
    model = ConvNeXt(
        in_chans=1,  # 输入通道数 (通常RGB图像为3)
        num_classes=49,  # 分类类别数，例如 ImageNet 数据集为 1000 类
        depths=[3, 3, 9, 3],  # 每个 stage 包含的 block 数量
        dims=[96, 192, 384, 768],  # 每个 stage 的通道数
        drop_path_rate=0.1,  # DropPath 概率
        layer_scale_init_value=1e-6,  # 层缩放的初始值
        head_init_scale=1.0  # 线性分类头的初始化缩放
    )
    # Step 2: 将模型移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Step 3: 打印模型摘要
    summary(model, (12, 1, 500))
