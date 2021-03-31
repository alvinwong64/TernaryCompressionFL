from .TNT import kernels_cluster
import torch.nn as nn
import torch.nn.functional as F
import torch


class KernelsCluster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return kernels_cluster(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
# class KernelsCluster2(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return kernels_cluster(weights_f=x, channel=False)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

    
class TNTConv2d(nn.Conv2d):
    def forward(self, x):
        w = KernelsCluster.apply(self.weight) # .to(self.weight.device)
        # print(w)
        y = self._conv_forward(x, w)
        return y
    
# class TNTFConv2d(nn.Conv2d):
#     def forward(self, x):
#         w = KernelsCluster2.apply(self.weight) # .to(self.weight.device)
#         # print(w)
#         y = self._conv_forward(x, w)
#         return y


class TNTLinear(nn.Linear):
    def forward(self, x):
        w = KernelsCluster.apply(self.weight)
        return F.linear(x, w, self.bias)