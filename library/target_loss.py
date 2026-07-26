import torch
import torch.optim

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class TargetLossOptimizer(torch.optim.Optimizer):
    """
    Dynamic step-size optimizer that drives loss toward target_loss.
    """
    def __init__(self, params, target_loss, 
                 min_step=1e-4, max_step=1.0, clip_norm=None, eps=0, weight_decay=0.1):
        defaults = dict()
        super().__init__(params, defaults)
        
        self.target_loss = target_loss
        self.min_step = min_step
        self.max_step = max_step
        self.clip_norm = clip_norm
        self.eps = eps
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure):
        loss = closure[0]
        target_loss = closure[1]
        loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
        if target_loss is None:
            target_loss = self.target_loss
        
        # Compute gradient L2 norm
        total_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()

        # Optional gradient clipping
        if self.clip_norm is not None and total_norm > self.clip_norm:
            clip_coef = self.clip_norm / (total_norm + self.eps)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            total_norm = self.clip_norm

        if total_norm <= self.eps:
            step_size = 0.0
        else:
            step_size = (loss_val - target_loss) / (total_norm + self.eps)
            step_size = max(self.min_step, min(self.max_step, step_size))
        # logger.info(f"step_size: {step_size}, loss_val: {loss_val}, target_loss: {target_loss}, total_norm: {total_norm}")

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    update = p.grad
                    if update.dtype in {torch.float16, torch.bfloat16}:
                        update = update.float()

                    update.mul_(step_size)

                    p_data_fp32 = p
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_data_fp32 = p_data_fp32.float()

                    if self.weight_decay != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=(-self.weight_decay * step_size))

                    p_data_fp32.add_(-update)

                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p.copy_(p_data_fp32)

        return step_size
