# Define network components here
import torch
from torch import nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act, 
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class R3LiteNet(torch.nn.Module):
    def __init__(self, in_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d,
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(R3LiteNet, self).__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        dilation_config = [1] * n_resblocks
        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)

        if pyramid:
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)

        self.deconv3 = ConvLayer(conv, n_feats, 6, kernel_size=1, stride=1, norm=None, act=act)
        self.residual_head = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        input_rgb = x[:, :3, :, :]

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.res_module(y)
        y = self.deconv1(y)
        y = self.deconv2(y)
        if self.pyramid_module is not None:
            y = self.pyramid_module(y)
        y = self.deconv3(y)

        output_t, output_r = torch.chunk(y, 2, dim=1)
        h = min(input_rgb.size(2), output_t.size(2), output_r.size(2))
        w = min(input_rgb.size(3), output_t.size(3), output_r.size(3))
        input_rgb = input_rgb[:, :, :h, :w]
        output_t = output_t[:, :, :h, :w]
        output_r = output_r[:, :, :h, :w]
        residual = self.residual_head(torch.cat([input_rgb, output_t, output_r], dim=1)) * 0.25
        return output_t, output_r, residual


class GatedR3LiteNet(torch.nn.Module):
    """Baseline-preserving R3Lite refiner.

    The backbone layer names intentionally match ERRNet/DRNet so an ERRNet
    checkpoint can initialize the baseline path through partial loading.
    """
    def __init__(self, in_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d,
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False, delta_scale=0.25):
        super(GatedR3LiteNet, self).__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.pyramid_module = None
        self.delta_scale = delta_scale

        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        dilation_config = [1] * n_resblocks
        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)

        if pyramid:
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)

        # Baseline path. This name/shape matches ERRNet and can be initialized
        # from checkpoints/errnet/errnet_060_00463920.pt.
        self.deconv3 = ConvLayer(conv, n_feats, 3, kernel_size=1, stride=1, norm=None, act=act)

        self.delta_head = nn.Sequential(
            nn.Conv2d(n_feats, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(n_feats, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.reflection_head = nn.Sequential(
            nn.Conv2d(n_feats, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def set_base_requires_grad(self, requires_grad):
        base_modules = [
            self.conv1, self.conv2, self.conv3, self.res_module,
            self.deconv1, self.deconv2, self.deconv3
        ]
        if self.pyramid_module is not None:
            base_modules.append(self.pyramid_module)
        for module in base_modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def init_gated_identity(self, mask_bias=-4.0):
        # Start from exactly the ERRNet baseline: T = T0 + M * 0.
        nn.init.zeros_(self.delta_head[-2].weight)
        nn.init.zeros_(self.delta_head[-2].bias)
        nn.init.zeros_(self.mask_head[-2].weight)
        nn.init.constant_(self.mask_head[-2].bias, mask_bias)
        nn.init.zeros_(self.reflection_head[-2].weight)
        nn.init.zeros_(self.reflection_head[-2].bias)
        nn.init.zeros_(self.residual_head[-2].weight)
        nn.init.zeros_(self.residual_head[-2].bias)

    def forward(self, x):
        input_rgb = x[:, :3, :, :]

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.res_module(y)
        y = self.deconv1(y)
        y = self.deconv2(y)
        if self.pyramid_module is not None:
            y = self.pyramid_module(y)

        base_t = torch.clamp(self.deconv3(y), 0, 1)
        delta_t = self.delta_head(y) * self.delta_scale
        mask = self.mask_head(y)
        output_r = self.reflection_head(y)

        h = min(input_rgb.size(2), base_t.size(2), delta_t.size(2), mask.size(2), output_r.size(2))
        w = min(input_rgb.size(3), base_t.size(3), delta_t.size(3), mask.size(3), output_r.size(3))
        input_rgb = input_rgb[:, :, :h, :w]
        base_t = base_t[:, :, :h, :w]
        delta_t = delta_t[:, :, :h, :w]
        mask = mask[:, :, :h, :w]
        output_r = output_r[:, :, :h, :w]

        output_t = torch.clamp(base_t + mask * delta_t, 0, 1)
        residual = self.residual_head(torch.cat([input_rgb, output_t, output_r], dim=1)) * 0.25
        return output_t, output_r, residual, base_t, mask, delta_t


class DualExpertFusionNet(torch.nn.Module):
    """Fuse two frozen reflection-removal experts with a trainable mask head."""
    requires_expert_paths = True

    def __init__(self, in_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d,
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False,
    expert0_inet='errnet', expert1_inet='errnet_r3lite'):
        super(DualExpertFusionNet, self).__init__()
        self.expert0_inet = expert0_inet
        self.expert1_inet = expert1_inet
        self.expert0 = self._build_expert(
            expert0_inet, in_channels, n_feats, n_resblocks, norm,
            se_reduction, res_scale, bottom_kernel_size, pyramid)
        self.expert1 = self._build_expert(
            expert1_inet, in_channels, n_feats, n_resblocks, norm,
            se_reduction, res_scale, bottom_kernel_size, pyramid)

        self.mask_head = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def _build_expert(name, in_channels, n_feats, n_resblocks, norm,
    se_reduction, res_scale, bottom_kernel_size, pyramid):
        if name == 'errnet':
            return DRNet(
                in_channels, 3, n_feats, n_resblocks,
                norm=norm, res_scale=res_scale, se_reduction=se_reduction,
                bottom_kernel_size=bottom_kernel_size, pyramid=pyramid)
        if name == 'errnet_r3lite':
            return R3LiteNet(
                in_channels, n_feats, n_resblocks,
                norm=norm, res_scale=res_scale, se_reduction=se_reduction,
                bottom_kernel_size=bottom_kernel_size, pyramid=pyramid)
        if name == 'errnet_r3lite_gated':
            return GatedR3LiteNet(
                in_channels, n_feats, n_resblocks,
                norm=norm, res_scale=res_scale, se_reduction=se_reduction,
                bottom_kernel_size=bottom_kernel_size, pyramid=pyramid)
        raise ValueError(
            'Unsupported expert inet: %s. Use errnet, errnet_r3lite, or errnet_r3lite_gated.' % name)

    def init_gated_identity(self, mask_bias=-4.0):
        nn.init.zeros_(self.mask_head[-2].weight)
        nn.init.constant_(self.mask_head[-2].bias, mask_bias)

    def set_base_requires_grad(self, requires_grad):
        for expert in (self.expert0, self.expert1):
            for param in expert.parameters():
                param.requires_grad = requires_grad

    @staticmethod
    def _state_icnn(state):
        return state['icnn'] if isinstance(state, dict) and 'icnn' in state else state

    def load_experts(self, expert0_state, expert1_state):
        self.expert0.load_state_dict(self._state_icnn(expert0_state))
        self.expert1.load_state_dict(self._state_icnn(expert1_state))
        self.set_base_requires_grad(False)

    @staticmethod
    def _split_output(output):
        if isinstance(output, (tuple, list)):
            t = output[0]
            r = output[1] if len(output) > 1 else torch.zeros_like(t)
            residual = output[2] if len(output) > 2 else torch.zeros_like(t)
            return t, r, residual
        return output, torch.zeros_like(output), torch.zeros_like(output)

    @staticmethod
    def _crop_like(tensors):
        h = min(t.size(2) for t in tensors)
        w = min(t.size(3) for t in tensors)
        return [t[:, :, :h, :w] for t in tensors]

    def forward(self, x):
        input_rgb = x[:, :3, :, :]

        with torch.no_grad():
            output0 = self.expert0(x)
            output1 = self.expert1(x)

        t0, _, _ = self._split_output(output0)
        t1, r1, residual1 = self._split_output(output1)
        input_rgb, t0, t1, r1, residual1 = self._crop_like([input_rgb, t0, t1, r1, residual1])
        t0 = torch.clamp(t0, 0, 1)
        t1 = torch.clamp(t1, 0, 1)

        mask_input = torch.cat([
            input_rgb,
            t0,
            t1,
            (t1 - t0).abs(),
            (input_rgb - t0).abs(),
            (input_rgb - t1).abs(),
        ], dim=1)
        mask = self.mask_head(mask_input)
        delta = t1 - t0
        output_t = torch.clamp(t0 + mask * delta, 0, 1)
        reconstruction = torch.clamp(output_t + r1 + residual1, 0, 1)
        residual = reconstruction - output_t - r1
        return output_t, r1, residual, t0, mask, delta


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
