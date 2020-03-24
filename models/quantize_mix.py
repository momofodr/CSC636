from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='extreme', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        # TODO: re-add true zero computation
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class my_clamp_round(InplaceFunction):

    @staticmethod
    def forward(ctx, input, min_value, max_value):
        ctx.input = input
        ctx.min = min_value
        ctx.max = max_value
        return torch.clamp(torch.round(input), min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        mask = (ctx.input > ctx.min) * (ctx.input < ctx.max)
        grad_input = mask.float() * grad_input
        return grad_input, None, None


class UniformQuantize():

    @staticmethod
    def quantize(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, prec_sf=None):

        output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits

        qmin = 0

        if prec_sf:
            prec = my_clamp_round().apply(prec_sf*num_bits,4,10)
        else:
            prec = num_bits
        qmax = qmin + 2.**prec - 1.

        scale = qparams.range / (qmax - qmin)

        output.add_(qmin * scale - zero_point).div_(scale)
        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        # quantize
        output = my_clamp_round().apply(output, qmin, int(qmax))

        if dequantize:
            output.mul_(scale).add_(
                zero_point - qmin * scale)  # dequantize
            
        return output



class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim, reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, prec_sf=None):
    return UniformQuantize.quantize(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace, prec_sf)


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True, signed=False, stochastic=True):
    return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace
        self.num_bits = num_bits

    def forward(self, input, bits, prec_sf=None, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=bits, flatten_dims=self.flatten_dims, reduce_dim=0)
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace, prec_sf=prec_sf)
            return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_bits=8, num_bits_weight=8):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.quantize_input = QuantMeasure(
            self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))

        self.weight_prec_sf = nn.Parameter(torch.tensor(1.))
        # self.act_prec_sf = nn.Parameter(torch.tensor(1.))
        self.act_prec_sf = None


    def forward(self, input, num_bits_weight=None, num_bits=None, ret_qinput=False):
        if num_bits_weight is None:
            num_bits_weight = self.num_bits_weight
        if num_bits is None:
            num_bits = self.num_bits

        if num_bits_weight == 0:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
            
        qinput = self.quantize_input(input, num_bits, self.act_prec_sf)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams, prec_sf=self.weight_prec_sf)

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=num_bits_weight,
                flatten_dims=(0, -1))
        else:
            qbias = None
        output = F.conv2d(qinput, qweight, qbias, self.stride,
                          self.padding, self.dilation, self.groups)
        if ret_qinput:
            return output, dict(type='conv', data=qinput.detach(),
                                kernel_size=self.kernel_size[0],
                                in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                stride=self.stride, padding=self.padding)
        else:
            return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True,
                 num_bits=8, num_bits_weight=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.quantize_input = QuantMeasure(self.num_bits)

        self.weight_prec_sf = nn.Parameter(torch.tensor(1.))
        # self.act_prec_sf = nn.Parameter(torch.tensor(1.))
        self.act_prec_sf = None


    def forward(self, input, ret_qinput=False):
        qinput = self.quantize_input(input, self.num_bits, self.act_prec_sf)
        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams, prec_sf=self.weight_prec_sf)
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits_weight,
                flatten_dims=(0, -1))
        else:
            qbias = None

        output = F.linear(qinput, qweight, qbias)
        if ret_qinput:
            return output, dict(type='fc', data=qinput.detach(),
                                in_features=self.in_features,
                                out_features=self.out_features)
        else:
            return output


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True,
                 num_chunks=16, eps=1e-5, num_bits=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x, self.num_bits)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, (B * H * W) // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = (mean_max - mean_min) * scale_fix
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    mean * (1 - self.momentum))

                self.running_var.mul_(self.momentum).add_(
                    scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        out = (x - mean.view(1, -1, 1, 1)) / \
            (scale.view(1, -1, 1, 1) + self.eps)

        if self.weight is not None:
            qweight = self.weight
            out = out * qweight.view(1, -1, 1, 1)

        if self.bias is not None:
            qbias = self.bias
            out = out + qbias.view(1, -1, 1, 1)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class RangeBN1d(RangeBN):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8):
        super(RangeBN1d, self).__init__(num_features, dim, momentum,
                                        affine, num_chunks, eps, num_bits)
        self.quantize_input = QuantMeasure(
            self.num_bits, inplace=True, shape_measure=(1, 1), flatten_dims=(1, -1))

if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)