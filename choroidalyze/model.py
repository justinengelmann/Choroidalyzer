import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding: [int, str] = 'same', conv_kwargs=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_kwargs=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, pool_layer=nn.MaxPool2d,
                 use_resid_connection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.pool = pool_layer(kernel_size=2)

        self.use_resid_connection = use_resid_connection
        if use_resid_connection:
            self.resid_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool(x)
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if self.use_resid_connection:
            x_out += self.resid_connection(x)
        return x_out

    def __repr__(self):
        return f'DownBlock({self.in_channels}->{self.out_channels})'


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_kwargs=None,
                 x_skip_channels=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, up_type='interpolate',
                 use_resid_connection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_skip_channels = x_skip_channels or in_channels // 2
        self.up_type = up_type
        if conv_kwargs is None:
            conv_kwargs = {}

        # upsample will double the number of channels, so we need to halve the number of channels in the input
        conv1_in_channels = in_channels // 2 + self.x_skip_channels

        self.conv1 = ConvNormAct(conv1_in_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        if up_type == 'interpolate':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                          nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1))
        elif up_type == 'convtranspose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        elif up_type == 'conv_then_interpolate':
            self.upsample = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
                                          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        else:
            raise ValueError(
                f'Unknown up_type: {up_type}, must be "interpolate", "convtranspose", "conv_then_interpolate"')

        self.use_resid_connection = use_resid_connection
        if use_resid_connection:
            self.resid_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if self.use_resid_connection:
            x_out += self.resid_connection(x)
        return x_out

    def __repr__(self):
        return f'UpBlock({self.up_type}, {self.in_channels}->{self.out_channels})'


class PadIfNecessary(nn.Module):
    """Pad input to make it divisible by 2^depth. Has .pad() and .unpad() methods"""

    # TODO: fix
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.two_to_depth = 2 ** depth
        self.pad_amt = None
        self.unpad_loc = None

    def get_pad_amt(self, x):
        b, c, h, w = x.shape
        pad_h = (self.two_to_depth - h % self.two_to_depth) % self.two_to_depth
        pad_w = (self.two_to_depth - w % self.two_to_depth) % self.two_to_depth
        # pad_amt = [pad_left, pad_right, pad_top, pad_bottom]
        pad_amt = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        return pad_amt

    def get_unpad_loc(self, x):
        b, c, h, w = x.shape
        # unpad wil deal with padded inputs, so we need to account for the padding here
        h += self.pad_amt[2] + self.pad_amt[3]
        w += self.pad_amt[0] + self.pad_amt[1]

        # all elements in batch, all channels, top to bottom, left to right
        unpad_loc = [slice(None), slice(None),
                     slice(self.pad_amt[2], h - self.pad_amt[3]),
                     slice(self.pad_amt[0], w - self.pad_amt[1])]
        return unpad_loc

    def pad(self, x):
        if self.pad_amt is None:
            self.pad_amt = self.get_pad_amt(x)
            self.unpad_loc = self.get_unpad_loc(x)
        return nn.functional.pad(x, self.pad_amt)

    def unpad(self, x):
        if self.pad_amt is None:
            raise ValueError('Must call .pad() before .unpad()')
        return x[self.unpad_loc]


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, channels: [int, str, list] = 32,
                 kernel_size=3, conv_kwargs=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, pool_layer=nn.MaxPool2d, up_type='interpolate',
                 extra_out_conv=False, use_resid_connection=False, dynamic_padding=False):
        super().__init__()

        self.depth = depth
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.pool_layer = pool_layer
        self.up_type = up_type
        self.extra_out_conv = extra_out_conv
        self.use_resid_connection = use_resid_connection

        if isinstance(channels, int):
            # default to doubling channels at each layer
            channels = [channels * 2 ** i for i in range(depth + 1)]
        elif isinstance(channels, str):
            initial_channels, strategy = channels.split('_')
            initial_channels = int(initial_channels)
            if strategy == 'double':
                channels = [initial_channels * 2 ** i for i in range(depth + 1)]
            elif strategy == 'same':
                channels = [initial_channels] * (depth + 1)
            elif strategy.startswith('doublemax'):
                max_channels = int(strategy.split('-')[1])
                channels = [min(initial_channels * 2 ** i, max_channels) for i in range(depth + 1)]
            else:
                raise ValueError(f'Unknown strategy: {strategy}')
        elif isinstance(channels, list):
            assert len(channels) == (depth + 1), f'channels must be a list of length {depth + 1}'

        self._unet_channels = channels

        if conv_kwargs is None:
            conv_kwargs = {}

        self.dynamic_padding = dynamic_padding
        if dynamic_padding:
            self.pad_if_necessary = PadIfNecessary(depth)

        self.in_conv = ConvNormAct(in_channels, channels[0], kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                   norm_layer=norm_layer, act_layer=act_layer)

        self.down_blocks = nn.ModuleList()
        for d in range(depth):
            self.down_blocks.append(DownBlock(channels[d], channels[d + 1], kernel_size, conv_kwargs=conv_kwargs,
                                              norm_layer=norm_layer, act_layer=act_layer, pool_layer=pool_layer,
                                              use_resid_connection=use_resid_connection))

        self.up_blocks = nn.ModuleList()
        for d in reversed(range(depth)):
            # self.up_blocks.append(UpBlock(channels[d + 1], channels[d], kernel_size, conv_kwargs=conv_kwargs,
            #                               norm_layer=norm_layer, act_layer=act_layer, up_type=up_type,
            #                               use_resid_connection=use_resid_connection))
            self.up_blocks.append(UpBlock(channels[d + 1], channels[d], kernel_size, conv_kwargs=conv_kwargs,
                                          x_skip_channels=channels[d],
                                          norm_layer=norm_layer, act_layer=act_layer, up_type=up_type,
                                          use_resid_connection=use_resid_connection))

        if not extra_out_conv:
            self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1)
        else:
            self.out_conv = nn.Sequential(
                ConvNormAct(channels[0], channels[0], kernel_size, padding=1, conv_kwargs=conv_kwargs,
                            norm_layer=norm_layer, act_layer=act_layer),
                nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1)
            )

    def forward(self, x):
        if self.dynamic_padding:
            x = self.pad_if_necessary.pad(x)

        x_skip = []
        x = self.in_conv(x)
        x_skip.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            x_skip.append(x)
        # remove last element of x_skip, which is the last output of the down_blocks
        x_skip.pop()

        for up_block in self.up_blocks:
            x = up_block(x, x_skip.pop())

        x = self.out_conv(x)

        if self.dynamic_padding:
            x = self.pad_if_necessary.unpad(x)
        return x