import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

loss = nn.CrossEntropyLoss()

def weighted_cross_entropy(outputs, target, alpha, reduction='sum'):
    """
    Spectral Norm Loss between one source and one target: ||output^T*output-target^T*target||_2
    Inputs:
        - output: torch.Tensor, source distribution
        - target: torch.Tensor, target distribution
    Output:
        - loss: float, value of the spectral norm of the difference of covariance matrix
        
    """
    if reduction == 'sum':
        weighted_loss = torch.sum(torch.stack([alpha[i]*loss(outputs[i], target[i]) for i in range(len(outputs))]))
    elif reduction == 'mean':
        weighted_loss = torch.mean(torch.stack([alpha[i]*loss(outputs[i], target[i]) for i in range(len(outputs))]))
    else:
        raise NotImplementedError
    
    return weighted_loss


def discrepancy(alpha, s_pred1, s_pred2, t_pred1, t_pred2):
    return torch.abs(target_disc(t_pred1, t_pred2) - source_disc(s_pred1, s_pred2, alpha))
    # return target_disc(t_pred1, t_pred2) - source_disc(s_pred1, s_pred2, alpha)

def target_disc(out1:list, out2:list):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def source_disc(out1:list, out2:list, alpha):
    return torch.sum(torch.stack([torch.mean(alpha[i]*torch.abs(F.softmax(out1[i]) - F.softmax(out2[i]))) for i in range(len(out1))]))


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)