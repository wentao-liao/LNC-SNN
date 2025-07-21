import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from surrogate import Rectangle
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


# class LIFSpike(nn.Module):
#     def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
#         super(LIFSpike, self).__init__()
#         self.act = ZIF.apply
#         # self.k = 10
#         # self.act = F.sigmoid
#         self.thresh = thresh
#         self.tau = tau
#         self.gama = gama
#
#     def forward(self, x):
#         mem = 0
#         spike_pot = []
#         T = x.shape[0]
#
#         for t in range(T):
#             mem = mem * self.tau + x[t,:]
#             spike = self.act(mem - self.thresh, self.gama)
#             # spike = self.act((mem - self.thresh)*self.k)
#             mem = (1 - spike) * mem
#             spike_pot.append(spike)
#         return torch.stack(spike_pot, dim=0)

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # 初始化可学习的阈值参数（H,W维度）
        self.thresh = nn.Parameter(torch.Tensor([thresh]).view(1, 1), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.Vth = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def forward(self, x):
        B, C, H, W = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # 扩展阈值到空间维度
        mem = torch.zeros(B, C, H, W, device=x.device)
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t]*self.alpha
            # 使用空间维度的阈值
            spike = self.act(mem - self.Vth, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=0)

# class LIFSpike(nn.Module):
#     def __init__(self, tau=2.0):
#         super(LIFSpike, self).__init__()
#         # the symbol is corresponding to the paper
#         # self.spike_func = surrogate_function
#         self.spike_func = Rectangle()
#         self.v_th = 1.
#         self.gamma = 1 - 1. / tau
#
#     def forward(self, x_seq):
#         # x_seq.shape should be [T, N, *]
#         _spike = []
#         u = 0
#         m = 0
#         T = x_seq.shape[0]
#         for t in range(T):
#             u = self.gamma * u + x_seq[t, ...]
#             spike = self.spike_func(u - self.v_th)
#             _spike.append(spike)
#             m = m * torch.sigmoid_((1. - self.gamma) * u) + spike
#             u = u - spike * (self.v_th + torch.sigmoid_(m))
#         # self.pre_spike_mem = torch.stack(_mem)
#         return torch.stack(_spike, dim=0)

# class LIFSpike(LIFNode_sj):
#     def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
#                  v_reset: float = None, surrogate_function = Rectangle(),
#                  detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
#         super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
#         self.register_memory('m', 0.)  # Complementary memory
#
#     def forward(self, x: torch.Tensor):
#         self.neuronal_charge(x)  # LIF charging
#         self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
#         spike = self.neuronal_fire()  # LIF fire
#         self.m += spike  # Strengthen
#         self.neuronal_reset(spike)  # LIF reset
#         self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
#         return spike
#
#     def neuronal_charge(self, x: torch.Tensor):
#         self._charging_v(x)
#
#     def neuronal_reset(self, spike: torch.Tensor):
#         self._reset(spike)
#
#     def _charging_v(self, x: torch.Tensor):
#         if self.decay_input:
#             x = x / self.tau
#
#         if self.v_reset is None or self.v_reset == 0:
#             if type(self.v) is float:
#                 self.v = x
#             else:
#                 self.v = self.v * (1 - 1. / self.tau) + x
#         else:
#             if type(self.v) is float:
#                 self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
#             else:
#                 self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x
#
#     def _reset(self, spike):
#         if self.v_reset is None:
#             # soft reset
#             self.v = self.v - spike * self.v_threshold
#         else:
#             # hard reset
#             self.v = (1. - spike) * self.v + spike * self.v_reset

# class LIFSpike(nn.Module):
#     def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
#         super(LIFSpike, self).__init__()
#         self.act = ZIF.apply
#         # 可学习的动态阈值参数（H,W维度）
#         self.thresh = nn.Parameter(torch.Tensor([thresh]).view(1, 1), requires_grad=True)
#         # 神经元连接矩阵（H,W维度）
#         self.connect_matrix = nn.Parameter(torch.ones(0, 0), requires_grad=True)
#         self.tau = tau
#         self.gama = gama
#
#     def forward(self, x):
#         B, C, H, W = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
#         mem = torch.zeros(B, C, H, W, device=x.device)
#         spike_pot = []
#         T = x.shape[0]
#
#         # 初始化连接矩阵（H,W）
#         if self.connect_matrix.shape[-2:] != (H, W):
#             self.connect_matrix = nn.Parameter(torch.eye(H * W, device=x.device).view(H, W, H, W), requires_grad=True)
#
#         for t in range(T):
#             # 膜电位衰减并加上输入
#             mem = mem * self.tau + x[t]
#
#             # 连接矩阵作用（空间维度交互）
#             mem = torch.einsum('bchw,hwkl->bckl', mem, self.connect_matrix)
#             # 动态阈值生成脉冲
#             spike = self.act(mem - self.thresh, self.gama)
#             # 膜电位重置
#             mem = (1 - spike) * mem
#
#             spike_pot.append(spike)
#
#         return torch.stack(spike_pot, dim=0)


if __name__ == "__main__":
    x=torch.randn([4, 64, 64, 32, 32])
    lif = DynamicConnectedLIF()
    y = lif(x)
    print(y.shape)
