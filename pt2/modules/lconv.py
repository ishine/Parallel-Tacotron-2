"""source: https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/transformer/lightconv.html#LightweightConvolution"""
"""Lightweight Convolution Module."""


import numpy
import torch
import torch.nn.functional as F
from torch import nn
MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class LConv(nn.Module):
    """Lightweight Convolution layer.
    This implementation is based on
    https://github.com/pytorch/fairseq/tree/master/fairseq

    Args:
        n_feat (int): the number of features
        wshare (int): the number of kernel of convolution (num. of heads)
        kernel_size (int): kernel size (length)
        dropout_rate (float): dropout_rate
        use_kernel_mask (bool): Use causal mask or not for convolution kernel
        use_bias (bool): Use bias term or not.

    """

    def __init__(
        self,
        n_feat,
        wshare,
        kernel_size,
        dropout_rate=0.1,
        use_kernel_mask=False,
        use_bias=False,
    ):
        """Construct Lightweight Convolution layer."""
        super(LConv, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.padding_size = int(kernel_size / 2)

        # lightconv related
        self.weight = nn.Parameter(
            torch.Tensor(self.wshare, 1, kernel_size).uniform_(0, 1)
        )
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))

        # mask of kernel
        kernel_mask0 = torch.zeros(self.wshare, int(kernel_size / 2))
        kernel_mask1 = torch.ones(self.wshare, int(kernel_size / 2 + 1))
        self.kernel_mask = torch.cat((kernel_mask1, kernel_mask0), dim=-1).unsqueeze(1)

    def forward(self, query, key, value, mask):
        """Forward of 'Lightweight Convolution'.

        This function takes query, key and value but uses only query.
        This is just for compatibility with self-attention layer (attention.py)

        Args:
            query (torch.Tensor): (batch, time1, d_model) input tensor
            key (torch.Tensor): (batch, time2, d_model) NOT USED
            value (torch.Tensor): (batch, time2, d_model) NOT USED
            mask (torch.Tensor): (batch, time1, time2) mask

        Return:
            x (torch.Tensor): (batch, time1, d_model) ouput

        """
        # linear -> GLU -> lightconv -> linear
        x = query
        B, T, C = x.size()
        H = self.wshare

        # lightconv
        x = x.transpose(1, 2).contiguous().view(-1, H, T)  # B x C x T
        weight = F.dropout(self.weight, self.dropout_rate, training=self.training)
        if self.use_kernel_mask:
            self.kernel_mask = self.kernel_mask.to(x.device)
            weight = weight.masked_fill(self.kernel_mask == 0.0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        x = F.conv1d(x, weight, padding=self.padding_size, groups=self.wshare).view(
            B, C, T
        )
        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)  # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        return x


class LConvBlock(torch.nn.Module):
    def __init__(self, dim, kernel_size, dropout_rate=0.1):
        super().__init__()

        self.layer_norm1 = torch.nn.LayerNorm(dim)
        self.glu_fc = torch.nn.Linear(dim, dim*2)
        self.glu = torch.nn.GLU()
        self.lconv = LConv(dim, 8, kernel_size, dropout_rate)
        self.layer_norm2 = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, dim*4)
        self.fc2 = torch.nn.Linear(dim*4, dim)

    def forward(self, x, mask=None):
        x_res = x
        x = self.layer_norm1(x)
        x = self.glu_fc(x)
        x = self.glu(x)
        x = self.lconv(x, None, None, mask=mask)
        x = x + x_res
        x_res = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + x_res

        return x
