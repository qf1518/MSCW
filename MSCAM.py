import torch
import torch.nn as nn
import einops
from MLA import MLA


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                 norm_type='bn', activation=True, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=use_bias)

        norm_class = nn.BatchNorm2d if norm_type == 'bn' else nn.GroupNorm
        self.norm = norm_class(32 if out_features >= 32 else out_features, out_features) if norm_type else None
        self.relu = nn.ReLU(inplace=False) if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.relu:
            x = self.relu(x)
        return x


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                 groups=None, norm_type='bn', activation=True, use_bias=True, pointwise=False):
        super().__init__()
        self.pointwise = pointwise
        self.depthwise = nn.Conv2d(in_channels=in_features, out_channels=in_features if pointwise else out_features,
                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation,
                                   bias=use_bias)

        if pointwise:
            self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                       bias=use_bias)

        norm_class = nn.BatchNorm2d if norm_type == 'bn' else nn.GroupNorm
        self.norm = norm_class(32 if out_features >= 32 else out_features, out_features) if norm_type else None
        self.relu = nn.ReLU(inplace=False) if activation else None

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm:
            x = self.norm(x)
        if self.relu:
            x = self.relu(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class UpsampleConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type=None, activation=False,
                 scale=(2, 2), conv='conv'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        conv_class = ConvBlock if conv == 'conv' else DepthwiseConvBlock
        self.conv = conv_class(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=(0, 0),
                               norm_type=norm_type, activation=activation)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DepthwiseProjection(nn.Module):
    def __init__(self, in_features, out_features, groups, kernel_size=(1, 1), padding=(0, 0), norm_type=None,
                 activation=False, pointwise=False):
        super().__init__()
        self.proj = DepthwiseConvBlock(in_features=in_features, out_features=out_features, kernel_size=kernel_size,
                                       padding=padding, groups=groups, pointwise=pointwise, norm_type=norm_type,
                                       activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        return einops.rearrange(x, 'B C H W -> B (H W) C')


class PoolEmbedding(nn.Module):
    def __init__(self, pooling, patch):
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        return einops.rearrange(x, 'B C H W -> B (H W) C')


class ScaleDotProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        att = self.softmax(torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale)
        return torch.einsum('bhcw, bhwk -> bhck', att, x3)


class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = DepthwiseProjection(in_features=out_features, out_features=out_features, groups=out_features)
        self.k_map = DepthwiseProjection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = DepthwiseProjection(in_features=in_features, out_features=in_features, groups=in_features)
        self.projection = DepthwiseProjection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = DepthwiseProjection(in_features=in_features, out_features=in_features, groups=in_features)
        self.k_map = DepthwiseProjection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = DepthwiseProjection(in_features=out_features, out_features=out_features, groups=out_features)
        self.projection = DepthwiseProjection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x


class CCSABlockWithLiteMLA(nn.Module):
    def __init__(self, features, out_channels=960, heads=4, scales=(3, 5)):
        super().__init__()
        self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
        self.lite_mla = MLA(in_channels=out_channels, out_channels=out_channels, heads=heads, scales=scales, dim=8,
                                kernel_func="relu")
        self.out_channels = out_channels
        self.features = features

    def forward(self, x):
        x_c = [self.channel_norm[i](xi) for i, xi in enumerate(x)]
        x_cin = torch.cat(x_c, dim=2)
        B, HW, total_C = x_cin.shape
        h = w = int(HW ** 0.5)
        x_cin = x_cin.transpose(1, 2).reshape(B, total_C, h, w)
        x_out = self.lite_mla(x_cin)
        x_out = x_out.flatten(2).transpose(1, 2)
        split_sizes = self.features
        x_out_split = torch.split(x_out, split_sizes, dim=2)
        return [x_out_split[i] + x[i] for i in range(len(x))]


class CCSABlock(nn.Module):
    def __init__(self, features, channel_head, spatial_head, spatial_att=True, channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.c_attention = nn.ModuleList([ChannelAttention(in_features=sum(features), out_features=feature, n_heads=head)
                                              for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.s_attention = nn.ModuleList([SpatialAttention(in_features=sum(features), out_features=feature, n_heads=head)
                                              for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        return self.m_apply(x_in, self.c_attention)

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        return self.m_apply(x_in, self.s_attention)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)


class MSCAM(nn.Module):
    def __init__(self, features, strides=[8, 4, 2, 1], patch=32, channel_att=True, spatial_att=True, n=1,
                 channel_head=[1, 2, 4, 4], spatial_head=[1, 2, 4, 4]):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(pooling=nn.AdaptiveAvgPool2d, patch=patch) for _ in features])
        self.avg_map = nn.ModuleList([DepthwiseProjection(in_features=feature, out_features=feature, kernel_size=(1, 1),
                                                           padding=(0, 0), groups=feature) for feature in features])
        self.attention = nn.ModuleList([CCSABlock(features=features, channel_head=channel_head, spatial_head=spatial_head,
                                                  channel_att=channel_att, spatial_att=spatial_att) for _ in range(n)])
        self.attention1 = CCSABlockWithLiteMLA(features=features, out_channels=960)
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, out_features=feature, kernel_size=(1, 1),
                                                   padding=(0, 0), norm_type=None, activation=False, scale=stride,
                                                   conv='conv') for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(feature), nn.ReLU()) for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = self.attention1(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        return self.m_apply(x_out, self.bn_relu)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)
