import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required


class RAD(Optimizer):
    r"""Implements relativistic adaptive gradient descent algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        delta (float, optional): strength of speed limitation (default: 1)
        order (int, optional): precision of the approximation to the relativistic Hamiltonian system (default: 1)
        eps(float, optional): the final value of increasing rational factor (default: 1)
        eps_annealing (int, optional): the iteration number for the rational factor increasing to the final value (default: 1)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), delta=1, order=1, eps=1, eps_annealing=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if order not in [1, 2]:
            raise ValueError("Invalid delta order: {}".format(order))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not eps_annealing >= 1:
            raise ValueError("Invalid eps_annealing value: {}".format(eps_annealing))
        
        defaults = dict(lr=lr, betas=betas, delta=delta, order=order, eps=eps, eps_annealing=eps_annealing)
        super(RAD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAD, self).__setstate__(state)

    def eps_annealing(self, x):
        eps = np.clip(math.exp(12 * math.pi * (x - 1)), 0, 1)
        return eps

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        kinetic_energy = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RAD does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                delta = group["delta"]
                lr = group["lr"]
                order = group["order"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                eps = group["eps"] * self.eps_annealing(state["step"] / group["eps_annealing"]) if group["eps_annealing"] != 1 else group["eps"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                denom = 1 / torch.sqrt(exp_avg_sq * (delta ** 2) * 4 + 4 * np.clip(eps * bias_correction2, 1e-16, 1))
                if order == 1:
                    denom *= 2
                elif order == 2:
                    denom += 1 / torch.sqrt(exp_avg_sq * (delta ** 2) * 4 + 4 * np.clip(eps * bias_correction2, 1e-16, 1) / (beta1 ** 2))
                denom *= math.sqrt(bias_correction2) / bias_correction1

                p.addcmul_(exp_avg, denom, value=-lr)

                kinetic_energy += lr / delta * torch.sum(torch.sqrt((exp_avg ** 2) / ((1 - beta1) ** 2) + 1 / (delta ** 2)))

        return loss, kinetic_energy

class SGD(Optimizer):
    r"""Implements stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0), when momentum is larger than 0, Heavy-ball (HB) methods are implemented
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, symplectic=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, symplectic=symplectic)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        exp_avg_norm_sq_total = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            symplectic = group["symplectic"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(d_p).detach()
                        if not symplectic:
                            d_p_tmp = buf
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        if not symplectic:
                            d_p_tmp = buf
                        buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                    if not symplectic:
                        d_p = d_p_tmp
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(d_p) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        return loss, kinetic_energy


class NAG(Optimizer):
    r"""Implements Nesterov’s accelerated gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(NAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAG, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        exp_avg_norm_sq_total = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                    d_p = buf.mul(momentum).add(d_p, alpha=1-momentum) / 2
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(buf) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        return loss, kinetic_energy


class DLPF(Optimizer):
    r"""Implements dissipative leapfrog method.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(DLPF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DLPF, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        exp_avg_norm_sq_total = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                    d_p = 1 / 2 * (momentum + 1) * buf
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(buf) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        return loss, kinetic_energy


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        kinetic_energy = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                denom = 1 / torch.sqrt(exp_avg_sq + group["eps"])
                denom *= math.sqrt(bias_correction2) / bias_correction1

                p.addcmul_(exp_avg, denom, value=-group["lr"])

                kinetic_energy += group["lr"] * torch.sum(torch.sqrt((exp_avg ** 2) / ((1 - beta1) ** 2) + 1))

        return loss, kinetic_energy


class RGD(Optimizer):
    r"""Implements relativistic gradient descent.

    RGD is based on the formula from
    `Conformal Symplectic and Relativistic Optimization`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0.9; 0 for standard SGD with lr = lr/2)
        delta (float, optional): strength of normalization (default: 1; 0 for a 2-order SGD-M method)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        order (int, optional): precision of the approximation to the relativistic Hamiltonian system (default: 1)
    """

    def __init__(self, params, lr=required, momentum=0.9, delta=1, weight_decay=0, order=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if order not in [1, 2]:
            raise ValueError("Invalid order: {}".format(order))

        defaults = dict(lr=lr, momentum=momentum, delta=delta, weight_decay=weight_decay)
        super(RGD, self).__init__(params, defaults)
        self.order = order

    def __setstate__(self, state):
        super(RGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        exp_avg_norm_sq_total = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            delta = group["delta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1-momentum)
                d_p = buf
                exp_avg_norm_sq = torch.norm(buf) ** 2
                lr_k = 1 / torch.sqrt((delta ** 2) * exp_avg_norm_sq + 1)
                if self.order == 1:
                    lr_k *= 2
                elif self.order == 2:
                    lr_k += 1 / torch.sqrt(
                        (delta ** 2) * exp_avg_norm_sq + 1 / (momentum ** 2)
                    )
                lr_k *= lr / 2
                p.add_(d_p, alpha=-lr_k)

                exp_avg_norm_sq_total += exp_avg_norm_sq

        kinetic_energy = lr / delta * torch.sqrt(exp_avg_norm_sq_total / ((1 - momentum) ** 2) + 1 / (delta ** 2))

        return loss, kinetic_energy
