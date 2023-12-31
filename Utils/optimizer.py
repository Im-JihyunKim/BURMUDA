import math
import torch
import torch.optim as optim
from torch.optim import Optimizer, SGD, AdamW, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR

class SGDW(Optimizer):
    def __init__(self,
                 params,
                 lr: float,
                 momentum: float = 0.,
                 dampening: float = 0.,
                 weight_decay: float = 0.,
                 nesterov: bool = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

        if nesterov and (momentum <= 0. or dampening != 0.):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening factor.")

        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                old = torch.clone(p.data).detach()
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)  # update current gradient with updated velocity
                    else:
                        d_p = buf                     # use current updated velocity as gradient

                p.data.add_(-group['lr'], d_p)

                if weight_decay != 0.:
                    p.data.add_(-weight_decay, old)

        return loss


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup & cosine decay.
       Implementation from `pytorch_transformers.optimization.WarmupCosineSchedule`.
       Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
       linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
       Decreases learning rate for 1. to 0. over remaining `t_total - warmup_steps` following a cosine curve.
       If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 t_total: int,
                 cycles: float = 0.5,
                 min_lr: float = 0.,
                 last_epoch: int = -1):
        
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        self.base_lr = optimizer.defaults['lr']
        
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        """A lambda function used as argument for `LambdaLR`."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
            mul = max(0., 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
            if self.expected_lr(mul) < self.min_lr:
                return self.min_lr / self.base_lr
            else:
                return mul

    def expected_lr(self, mul: float):
        return self.base_lr * mul


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Implementation from `pytorch_transformers.optimization.WarmupCosineWithHardRestartsSchedule`.
        Assuming that the initial learning rate of `optimizer` is set to 1., this scheduler
        linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        """A lambda function used as argument for `LambdaLR`."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


def get_optimizer(params, name: str, lr: float, weight_decay: float = 0.00, **kwargs):
    """Returns an `Optimizer` object given proper arguments."""

    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif name == 'adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'lookahead':
        raise NotImplementedError
    else:
        raise NotImplementedError

def get_scheduler(optimizer: optim.Optimizer, name: str, epochs: int, **kwargs):
    """Configure learning rate scheduler."""
    if name == 'step':
        step_size = kwargs.get('milestone', epochs // 10 * 9)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'cosine':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        return WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs)
    elif name == 'restart':
        warmup_steps = kwargs.get('warmup_steps', epochs // 10)
        cycles = kwargs.get('cycles', 4)
        return WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs, cycles=cycles)
    else:
        return None


def get_multi_step_scheduler(optimizer: optim.Optimizer, milestones: list, gamma: float = 0.1):
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def get_cosine_scheduler(optimizer: optim.Optimizer,
                         epochs: int,
                         warmup_steps: int = 0,
                         cycles: int = 1,
                         min_lr: float = 5e-3):
    """Configure half cosine learning rate schduler."""
    if warmup_steps < 0:
        return None
    if cycles <= 1:
        return WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs, min_lr=min_lr)
    else:
        return WarmupCosineWithHardRestartsSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs, cycles=cycles)