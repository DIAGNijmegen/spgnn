import torch.nn as nn
import torch
import numpy as np
import functools
import torch.nn.functional as F
from torch.nn import LayerNorm, InstanceNorm3d
import math

from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def shift_and_stitch_5d(model, x, patch_size, stride, out_ch):
    if np.prod(stride) == 1:
        return x
    pad_D, pad_W, pad_H = [p//2 for p in patch_size]
    stride_D, stride_W, stride_H = stride
    N, C, D, W, H = x.shape

    target_shape = N, C, D + 2 * pad_D, W + 2 * pad_W, H + 2 * pad_H
    y = torch.zeros((N, out_ch, D + stride_D, W + stride_W, H + stride_H)).type(x.type())
    for d in range(stride_D):
        for w in range(stride_H):
            for h in range(stride_W):
                x_in = torch.zeros(target_shape).type(x.type())
                d_start, w_start, h_start = pad_D - d, pad_W  - w, pad_H - h
                x_in[:, :, d_start:d_start + D, w_start: w_start + W, h_start: h_start + H] = x
                y_out = model(x_in)
                _, _, d_out, w_out, h_out = y_out.shape
                y[:, :, d:d_out*stride_D:stride_D,
                    w:w_out*stride_W:stride_W,
                    h:h_out*stride_H:stride_H] = y_out
    return y[:,:, :D, :W, :H]

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, args=None):
        return x


class L2NormLayer(nn.Module):
    """ simple implementation of l2 normalization layer of ParseNet. """

    def __init__(self, in_ch):
        super(L2NormLayer, self).__init__()
        self.rescale_param = nn.Parameter(torch.ones(1, in_ch, 1, 1, 1).fill_(10.0))
        self.affine_param = nn.Parameter(torch.ones(1, in_ch, 1, 1, 1).fill_(0.0))

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        x_flat_norm = torch.norm(x_flat, p=2, dim=1).unsqueeze(1)
        normalized_x = (x_flat / torch.clamp(torch.sqrt(x_flat_norm), min=0.0001)).view(x.shape)
        o = normalized_x * self.rescale_param + self.affine_param
        assert (o.shape == x.shape)
        return o


def normal_wrapper(normal_method, in_ch, in_ch_div=2):
    if normal_method is "bn":
        return nn.BatchNorm3d(in_ch)
    elif normal_method is "bnt":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=True, track_running_stats=False)
    elif normal_method is "bntna":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=False, track_running_stats=False)
    elif normal_method is "ln":
        return nn.GroupNorm(1, in_ch)
    elif normal_method is "lnna":
        return nn.GroupNorm(1, in_ch, affine=False)
    elif normal_method is "in":
        return nn.GroupNorm(in_ch, in_ch)
    elif normal_method is "in3dn":
        return InstanceNorm3d(num_features=in_ch, affine=True, track_running_stats=False)
    elif normal_method is "l2n":
        return L2NormLayer(in_ch)
    elif normal_method is "sbn":
        return nn.SyncBatchNorm(in_ch)
    else:
        return Identity()


def act_wrapper(act_method, num_parameters=1, init=0.25):
    if act_method is "relu":
        return nn.ReLU(inplace=True)
    elif act_method is "prelu":
        return nn.PReLU(num_parameters, init)
    else:
        raise NotImplementedError


def checkpoint_wrapper(module, segments, *tensors):
    if segments > 0:
        # if type(module) in [nn.Sequential, nn.ModuleList, list]:
        #     return checkpoint_sequential(module, segments, *tensors)
        # else:
        return checkpoint(module, *tensors)
    else:
        return module(*tensors)


def crop_concat_5d(t1, t2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    assert (t1.dim() == t2.dim() == 5)
    assert (t1.shape[-1] <= t2.shape[-1])
    slices = (slice(None, None), slice(None, None)) \
             + tuple(
        [slice(int(np.ceil((b - a) / 2)), a + int(np.ceil((b - a) / 2))) for a, b in zip(t1.shape[2:], t2.shape[2:])])
    x = torch.cat([t1, t2[slices]], dim=1)
    return x


def crop_concat_5d_with_pool(t1, t2, stride1, stride2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    assert (t1.dim() == t2.dim() == 5)
    assert (stride1 >= stride2)
    assert (t1.shape[-1] <= t2.shape[-1])
    pool_scale = stride1 // stride2
    slices = (slice(None, None), slice(None, None)) + \
             tuple([slice((ts2 - ts1 * pool_scale) // 2, ts2 - (ts2 - ts1 * pool_scale) // 2, pool_scale)
                    for ts1, ts2 in zip(t1.shape[-3:], t2.shape[-3:])])

    x = torch.cat([t1, t2[slices]], dim=1)
    return x


def crop_add_5d(t1, t2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    # print("t1:{} t2:{}".format(t1.shape, t2.shape))
    assert (t1.dim() == t2.dim() == 5)
    assert (t1.shape[-1] <= t2.shape[-1])
    assert (t1.shape[1] == t2.shape[1])
    slices = (slice(None, None), slice(None, None)) \
             + tuple([slice((b - a) // 2, a + (b - a) // 2) for a, b in zip(t1.shape[2:], t2.shape[2:])])
    x = t1 + t2[slices]
    return x


def conv3d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    if len(input.shape) != 5:
        raise AttributeError("support 5d tensor only.")
    if isinstance(dilation, int):
        dilation = [dilation] * 3
    if isinstance(stride, int):
        stride = [stride] * 3
    odds = []
    conv_pads = []
    for dim_idx, dim in enumerate(input.shape):
        filter_rows = weight.size(2 + dim_idx)
        out_rows = (dim + stride[dim_idx] - 1) // stride[dim_idx]
        padding_rows = max(0, (out_rows - 1) * stride[dim_idx] +
                           (filter_rows - 1) * dilation[dim_idx] + 1 - dim)
        odd = (padding_rows % 2 != 0)
        odds.append(odd)
        conv_pads.append(padding_rows)
    if any(odds):
        input = F.pad(input, np.insert(np.zeros(3), [1, 2, 3], odds, axis=0)
                      .astype(np.uint8).tolist())

    return F.conv3d(input, weight, bias, stride,
                    padding=tuple([padding_rows for padding_rows in conv_pads]),
                    dilation=dilation, groups=groups)


class SEBlock(nn.Module):

    def __init__(self, in_ch, out_ch, norm_method='bn', act_method='relu', reduction=2):
        super(SEBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.reduction = reduction
        if self.in_ch != self.out_ch:
            self.f_tr = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
                normal_wrapper(norm_method, out_ch),
                act_wrapper(act_method)
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.ex = nn.Sequential(
            nn.Linear(out_ch, out_ch // self.reduction, bias=False),
            act_wrapper(act_method),
            nn.Linear(out_ch // self.reduction, out_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.in_ch != self.out_ch:
            x = self.f_tr(x)
        b, c, *args = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.ex(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class RevResChainASPPBlock(nn.Module):

    def __init__(self, in_ch, chain_k_sizes, aspp_ksize, aspp_paddings,
                 aspp_rates, aspp_only_parallel,
                 chain_rates_list=[[1, 2, 3], [1, 2, 3]], norm_method='bn', act_method='relu',
                 is_act=True):
        super(RevResChainASPPBlock, self).__init__()
        self.is_act = is_act
        self.in_ch = in_ch
        self.chain_modules = nn.Sequential(*[RevChainAtrousConv(self.in_ch, k_size, rates)
                                             for k_size, rates in zip(chain_k_sizes, chain_rates_list)],
                                           ASPP5d(in_ch, in_ch, in_ch,
                                                  aspp_ksize, aspp_paddings, aspp_only_parallel,
                                                  aspp_rates)
                                           )
        self.act = act_wrapper(act_method)

    def forward(self, x):
        residual = x
        x = self.chain_modules(x)
        x = x + residual
        if self.is_act:
            x = self.act(x)
        return x



class ConvPoolBlock5d(nn.Module):

    def __init__(self, in_ch_list, base_ch_list, checkpoint_segments,
                 conv_ksize, conv_bias, conv_pad,
                 pool_ksize, pool_strides, pool_pad, dropout=0.1,
                 conv_strdes=1, norm_method='bn', act_method="relu",
                 **kwargs):
        super(ConvPoolBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_ch_list)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_ch_list)

        if not isinstance(conv_strdes, (tuple, list)):
            conv_strdes = [conv_strdes] * len(in_ch_list)

        if dropout > 0:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], stride=conv_strdes[idx],
                                          padding=conv_pad[idx], bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method),
                                nn.Dropout(dropout))
                  for idx, (in_ch, base_ch) in enumerate(zip(in_ch_list, base_ch_list))])
        else:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], stride=conv_strdes[idx],
                                          padding=conv_pad[idx], bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method)
                                )
                  for idx, (in_ch, base_ch) in enumerate(zip(in_ch_list, base_ch_list))])
        self.maxpool = nn.MaxPool3d(kernel_size=pool_ksize, stride=pool_strides, padding=pool_pad)

    def forward(self, x, args=None):
        y = self.conv_blocks(x)
        pooled = self.maxpool(y)
        return y, pooled


class UpsampleConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, scale_factor,
                 conv_ksize, conv_bias, conv_pad, dropout=0.1,
                 norm_method='bn', act_methpd='relu', **kwargs):
        super(UpsampleConvBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        self.scale_factor = scale_factor
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_chs)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_chs)

        if dropout > 0:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                    nn.Dropout(dropout)
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])
        else:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx], bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                ) for idx, (in_ch, base_ch) in enumerate(zip(in_chs, base_chs))
            ])

        self.merge_func = kwargs.get('merge_func', crop_concat_5d)
        self.upsample = nn.Upsample(size=None, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)

    def forward(self, inputs, cats, args=None):
        up_inputs = self.upsample(inputs)
        x = crop_concat_5d(up_inputs, cats)
        x = self.conv_blocks(x)
        return x


class ConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, conv_ksize,
                 conv_bias, conv_pad, dropout=0.1, conv_strides=1,
                 norm_method='bn', act_methpd='relu', lite=False,
                 **kwargs):
        super(ConvBlock5d, self).__init__()
        if not isinstance(conv_ksize, (tuple, list)):
            conv_ksize = [conv_ksize] * len(in_chs)

        if not isinstance(conv_pad, (tuple, list)):
            conv_pad = [conv_pad] * len(in_chs)

        if not isinstance(conv_strides, (tuple, list)):
            conv_strides = [conv_strides] * len(in_chs)

        if lite:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                              bias=conv_bias, stride=conv_stride),
                    act_wrapper(act_methpd),
                ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
            ])
        else:
            if dropout > 0:
                print("use dropout in convs!")
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                        nn.Dropout(dropout),
                    ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
                ])
            else:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize[idx], padding=conv_pad[idx],
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                    ) for idx, (in_ch, base_ch, conv_stride) in enumerate(zip(in_chs, base_chs, conv_strides))
                ])

    def forward(self, x, args=None):
        return self.conv_blocks(x)


class DeConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, conv_ksizes,
                 conv_bias, conv_pads, conv_out_paddings, dropout=0.1, conv_strides=[1, 1],
                 norm_method='bn', act_methpd='relu', lite=False,
                 **kwargs):
        super(DeConvBlock5d, self).__init__()
        if lite:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, base_ch, kernel_size=k_size, padding=conv_pad,
                                       bias=conv_bias, stride=conv_stride, output_padding=conv_out_padding),
                    act_wrapper(act_methpd),
                ) for in_ch, base_ch, conv_stride, conv_out_padding, k_size, conv_pad in
                zip(in_chs, base_chs, conv_strides, conv_out_paddings, conv_ksizes, conv_pads)
            ])
        else:
            if dropout > 0:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.ConvTranspose3d(in_ch, base_ch, kernel_size=k_size, padding=conv_pad,
                                           bias=conv_bias, stride=conv_stride, output_padding=conv_out_padding),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                        nn.Dropout(dropout),
                    ) for in_ch, base_ch, conv_stride, conv_out_padding, k_size, conv_pad in
                    zip(in_chs, base_chs, conv_strides, conv_out_paddings, conv_ksizes, conv_pads)
                ])
            else:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.ConvTranspose3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad,
                                           bias=conv_bias, stride=conv_stride,
                                           output_padding=conv_out_padding),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                    ) for in_ch, base_ch, conv_stride, conv_out_padding, conv_ksize, conv_pad in
                    zip(in_chs, base_chs, conv_strides, conv_out_paddings, conv_ksizes, conv_pads)
                ])

    def forward(self, x, args=None):
        return self.conv_blocks(x)


class ResConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments,
                 conv_ksize, conv_bias, conv_pad, conv_strides,
                 dropout=0.1, norm_method='bn', act_method='relu',
                 **kwargs):
        super(ResConvBlock5d, self).__init__()
        self.res_head = nn.Sequential(
            nn.Conv3d(in_chs[0], base_chs[0], kernel_size=conv_ksize, padding=conv_pad,
                      bias=conv_bias, stride=conv_strides[0]),
            normal_wrapper(norm_method, base_chs[0]),
        )
        res_middle = [
            nn.Sequential(
                nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad,
                          bias=conv_bias, stride=conv_stride),
                normal_wrapper(norm_method, base_ch),
                act_wrapper(act_method),
                nn.Dropout(dropout),
            ) for in_ch, base_ch, conv_stride in zip(in_chs[1:-1], base_chs[1:-1], conv_strides[1:-1])
        ]
        res_end = [
            nn.Sequential(
                nn.Conv3d(in_chs[-1], base_chs[-1], kernel_size=conv_ksize, padding=conv_pad, bias=conv_bias,
                          stride=conv_strides[-1]),
                normal_wrapper(norm_method, base_chs[-1]),
            )
        ]
        self.act = act_wrapper(act_method)
        self.drop = nn.Dropout(dropout)
        self.conv_blocks = nn.Sequential(*(res_middle + res_end))
        self.checkpoint_segments = checkpoint_segments

    def forward(self, x, args=None):
        res = self.res_head(x)
        out = checkpoint(self.conv_blocks, res)
        merged = crop_add_5d(out, res)
        out = self.act(merged)
        return self.drop(out)


class ChainAtrousConv(nn.Module):

    def __init__(self, in_ch, k_size=[3, 3, 3], rates=[1, 2, 5], padding=[1, 2, 5], drop_out=0.1,
                 norm_method='bn', act_method='relu'):
        super(ChainAtrousConv, self).__init__()
        self.rates = rates
        self.k_size = k_size
        self.padding = padding
        self.in_ch = in_ch
        self.convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=ksize,
                          dilation=rate, padding=pad, bias=False),
                normal_wrapper(norm_method, in_ch),
                act_wrapper(act_method),
                nn.Dropout(drop_out)
            ) for rate, ksize, pad in zip(self.rates, self.k_size, self.padding)])

    def forward(self, x):
        return self.convs(x)


class ResChainAtrousConv(nn.Module):

    def __init__(self, in_ch, out_ch, k_sizes=[3, 3, 3], rates=[1, 2, 5],
                 norm_method='bn', act_method='relu'):
        super(ResChainAtrousConv, self).__init__()
        self.rates = rates
        self.k_sizes = k_sizes
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convs = nn.Sequential(*([nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=ksize,
                      dilation=rate, padding=rate, bias=False),
            normal_wrapper(norm_method, in_ch),
        ) for rate, ksize in
                                         zip(self.rates[:-1], self.k_sizes[:-1])] +
                                     [nn.Sequential(
                                         nn.Conv3d(in_ch, out_ch, kernel_size=self.k_sizes[-1],
                                                   dilation=self.rates[-1], padding=self.rates[-1], bias=False),
                                         normal_wrapper(norm_method, out_ch),
                                     )]))
        self.reshape = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                                     normal_wrapper(norm_method, out_ch),
                                     )
        self.relu = act_wrapper(act_method)

    def forward(self, x):
        r = self.reshape(x)
        y = self.convs(x)
        return F.relu(r + y, inplace=True)


class ChainAtrousConvReshape(nn.Module):

    def __init__(self, in_ch, out_ch, k_size=3, rates=[1, 2, 5],
                 norm_method='bn', act_method='relu'):
        super(ChainAtrousConvReshape, self).__init__()
        self.rates = rates
        self.in_ch = in_ch
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=k_size, padding=rate, dilation=rate, bias=False),
            normal_wrapper(norm_method, in_ch),
            act_wrapper(act_method)) for rate in self.rates[:-1]])
        self.f_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=k_size, padding=rates[-1], dilation=rates[-1], bias=False),
            normal_wrapper(norm_method, in_ch))
        self.relu = act_wrapper(act_method)
        self.re_shape = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            normal_wrapper(norm_method, out_ch),
            act_wrapper(act_method))

    def forward(self, x):
        residual = x
        for conv in self.convs:
            x = conv(x)

        x = self.f_conv(x)
        x = x + residual
        x = self.relu(x)
        x = self.re_shape(x)
        return x


class ASPP5d(nn.Module):
    def __init__(self, in_ch, base_ch, out_ch,
                 norm_method='bn', act_method='relu',
                 dilations=[2, 3, 5]):
        super(ASPP5d, self).__init__()
        self.global_averge_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                                nn.Conv3d(in_ch, base_ch, kernel_size=1, stride=1,
                                                          padding=0, bias=False),
                                                normal_wrapper(norm_method, base_ch),
                                                act_wrapper(act_method),
                                                )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=1, padding=0, dilation=1, bias=False),
            normal_wrapper(norm_method, base_ch),
            act_wrapper(act_method),
        )

        self.aspp_convs = nn.ModuleList([])
        for dilation in zip(dilations):
            conv_c = nn.Sequential(
                nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=dilation,
                          dilation=dilation, bias=False),
                normal_wrapper(norm_method, base_ch),
                act_wrapper(act_method),
            )
            self.aspp_convs.append(conv_c)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_ch * (2 + len(dilations)), out_ch,
                      kernel_size=1, padding=0, dilation=1,
                      bias=False),
            normal_wrapper(norm_method, out_ch),
            act_wrapper(act_method),
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feat1 = F.interpolate(self.global_averge_pool(x), size=x.size()[2:], mode='trilinear', align_corners=True)
        feat2 = self.conv2(x)
        aspp_feat = [feat1, feat2]
        for aspp_conv in self.aspp_convs:
            feat = aspp_conv(x)
            aspp_feat.append(feat)
        out = torch.cat(aspp_feat, 1)

        bottle = self.bottleneck(out)
        return bottle


class IRNNlayer(nn.Module):

    def __init__(self, in_ch, hidden_ch, rnn_dir, norm_method, act_method, padding_mode='same', drop_rate=0.1):
        super(IRNNlayer, self).__init__()
        self.rnn_dir = rnn_dir
        self.padding_mode = padding_mode
        self.w_x = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=3, bias=False, padding=1) if self.padding_mode == 'same' else
            nn.Conv3d(in_ch, hidden_ch, kernel_size=1, bias=False, padding=0),
            normal_wrapper(norm_method, hidden_ch),
            act_wrapper(act_method)
        )
        self.rnns = nn.ModuleList([ConvRNNCell3d(in_ch, hidden_ch, self.w_x, self.padding_mode,
                                                 norm_method=norm_method, act_method=act_method,
                                                 seq_dim=int(x // 2) + 2, reverse=x % 2 == 0) for x in range(6)])
        self.dropout = nn.Dropout(drop_rate)
        self.reshape = nn.Sequential(
            nn.Conv3d(hidden_ch * self.rnn_dir, hidden_ch,
                      kernel_size=1, bias=False),
            normal_wrapper(norm_method, hidden_ch),
            act_wrapper(act_method)
        )
        for m in self.modules():
            # he kaiming's init
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(1, math.sqrt(2. / n))

    def forward(self, x):
        i_x = torch.cat([rnn(x) for rnn in self.rnns], dim=1)
        i_x = self.dropout(i_x)
        return self.reshape(i_x)


class ConvRNNCell3d(nn.Module):

    def __init__(self, in_ch, hidden_ch, x_2_h_module, padding_mode, norm_method='l2n', act_method='relu',
                 seq_dim=4, reverse=False):
        super(ConvRNNCell3d, self).__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.seq_dim = seq_dim
        self.reverse = reverse
        self.x_2_h_module = x_2_h_module
        self.padding_mode = padding_mode
        self.w_h = nn.Sequential(
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=(1, 3, 3), bias=False,
                      padding=(0, 1, 1)) if padding_mode == 'same' else
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            normal_wrapper(norm_method, hidden_ch),
        )
        self.act = act_wrapper(act_method)

    # def step(self, x_slice, h):
    #     h = self.w_x(x_slice.unsqueeze(2)) + self.w_h(h)
    #     return h
    #
    # def forward(self, x):
    #     if self.seq_dim != 2:
    #         x = x.transpose(self.seq_dim, 2).contiguous()
    #
    #     h = x.new(x.shape[0], self.hidden_ch, 1, *x.shape[3:])
    #     if x.is_cuda and torch.cuda.is_available():
    #         h = h.cuda()
    #     _iter = reversed(range(x.shape[2])) if self.reverse else range(x.shape[2])
    #
    #     o = []
    #     for x_slice_id in _iter:
    #         h = self.step(x[:, :, x_slice_id, ::], h)
    #         o.append(h)
    #
    #     return torch.cat(o, dim=2).transpose(2, self.seq_dim).contiguous()

    def step(self, x_hidden_slice, h):
        h = self.act(x_hidden_slice.unsqueeze(2) + self.w_h(h))
        return h

    def forward(self, x):
        x_hidden = self.x_2_h_module(x)
        if self.seq_dim != 2:
            x_hidden = x_hidden.transpose(self.seq_dim, 2).contiguous()

        h = x.new(x_hidden.shape[0], self.hidden_ch, 1, *x_hidden.shape[3:]).zero_()
        if x.is_cuda and torch.cuda.is_available():
            h = h.cuda()
        _iter = reversed(range(x_hidden.shape[2])) if self.reverse else range(x_hidden.shape[2])

        o = []
        for x_slice_id in _iter:
            h = self.step(x_hidden[:, :, x_slice_id, ::], h)
            o.append(h)

        return torch.cat(o, dim=2).transpose(2, self.seq_dim).contiguous()
