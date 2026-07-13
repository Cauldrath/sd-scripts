import torch
import torch.optim

# from library.utils import setup_logging

# setup_logging()
# import logging

# logger = logging.getLogger(__name__)


class TargetLossOptimizer(torch.optim.Optimizer):
    """
    Dynamic step-size optimizer that drives loss toward target_loss.
    Preserves base optimizer state while making the effective LR irrelevant.
    Compatible with SGD, Adam, AdamW, Adafactor, and other torch.optim optimizers.
    """
    def __init__(self, params, base_optimizer, target_loss, 
                 min_step=1e-4, max_step=1.0, clip_norm=None, eps=1e-8):
        defaults = dict()
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.target_loss = target_loss
        self.min_step = min_step
        self.max_step = max_step
        self.clip_norm = clip_norm
        self.eps = eps

    def step(self, loss):
        loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
        
        # 1. Compute gradient L2 norm
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # 2. Optional gradient clipping
        if self.clip_norm is not None and total_norm > self.clip_norm:
            clip_coef = self.clip_norm / (total_norm + self.eps)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            total_norm = self.clip_norm

        # 3. Compute target step size (effective LR) via Taylor approximation
        if total_norm < self.eps:
            step_size = 0.0
        else:
            step_size = (loss_val - self.target_loss) / (total_norm ** 2 + self.eps)
            step_size = max(self.min_step, min(self.max_step, step_size))
        # logger.info(f"step_size: {step_size}")

        # 4. Temporarily set LR to target step size, step, then restore
        saved_lrs = []
        for group in self.param_groups:
            saved_lrs.append(group['lr'])
            group['lr'] = step_size
        
        self.base_optimizer.step()
        
        for group, orig_lr in zip(self.param_groups, saved_lrs):
            group['lr'] = orig_lr

        return step_size

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)