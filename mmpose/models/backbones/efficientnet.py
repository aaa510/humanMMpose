import torch
import torch.nn as nn
import math
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmpose.registry import MODELS

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class MBConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_se = se_ratio > 0
        self.use_residual = input_channels == output_channels and stride == 1

        # Expansion phase
        expanded_channels = input_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(input_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = None

        # Depthwise convolution phase
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )

        # Squeeze and excitation phase
        if self.use_se:
            squeeze_channels = max(1, int(input_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, squeeze_channels)

        # Output phase
        self.output_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.output_conv(x)
        if self.use_residual:
            x = x + x
        return x

@MODELS.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone for MMPose.
    
    Args:
        arch (str): Architecture of EfficientNet, choices are ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        in_channels (int): Number of input channels. Default: 3
        out_indices (tuple): Output from which stages. Default: (0, 1, 2, 3, 4)
        frozen_stages (int): Stages to be frozen (all param fixed). Default: -1
        conv_cfg (dict): Config dict for convolution layer. Default: None
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN')
        norm_eval (bool): Whether to set norm layers to eval mode. Default: True
        with_cp (bool): Use checkpoint or not. Default: False
    """

    arch_settings = {
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
    }

    def __init__(self,
                 arch='b0',
                 in_channels=3,
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False):
        super(EfficientNet, self).__init__()
        if arch not in self.arch_settings:
            raise KeyError(f'invalid arch {arch}')
        
        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        # Get architecture parameters
        width_mult, depth_mult, _, dropout_rate = self.arch_settings[arch]
        
        # Initial convolution
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, _make_divisible(32 * width_mult, 8), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_make_divisible(32 * width_mult, 8)),
            nn.SiLU(inplace=True)
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.stage_names = []
        
        # Stage configurations
        stage_configs = [
            # t, c, n, s, k, se
            [1, 16, 1, 1, 3, 0.25],
            [6, 24, 2, 2, 3, 0.25],
            [6, 40, 2, 2, 5, 0.25],
            [6, 80, 3, 2, 3, 0.25],
            [6, 112, 3, 1, 5, 0.25],
            [6, 192, 4, 2, 5, 0.25],
            [6, 320, 1, 1, 3, 0.25],
        ]

        # Build each stage
        input_channels = _make_divisible(32 * width_mult, 8)
        for i, (t, c, n, s, k, se) in enumerate(stage_configs):
            output_channels = _make_divisible(c * width_mult, 8)
            stage = []
            for j in range(n):
                stride = s if j == 0 else 1
                stage.append(MBConv(
                    input_channels if j == 0 else output_channels,
                    output_channels,
                    k,
                    stride,
                    t,
                    se
                ))
                input_channels = output_channels
            self.stages.append(nn.Sequential(*stage))
            self.stage_names.append(f'stage{i+1}')

        # Final convolution
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channels, _make_divisible(1280 * width_mult, 8), 1, bias=False),
            nn.BatchNorm2d(_make_divisible(1280 * width_mult, 8)),
            nn.SiLU(inplace=True)
        )

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv_stem.eval()
            for param in self.conv_stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'stage{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained))
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_stem(x)
        
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        
        x = self.conv_head(x)
        if len(self.stages) in self.out_indices:
            outs.append(x)
            
        return tuple(outs)

    def train(self, mode=True):
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval() 