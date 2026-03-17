import torch
import torch.nn as nn 

class MultiTaskLoss(nn.Module):
    def __init__(self, task1_loss_fn, task2_loss_fn, is_regression):
        super().__init__()
        self.task1_loss_fn = task1_loss_fn
        self.task2_loss_fn = task2_loss_fn
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, task1_pred, task2_pred, task1_target, task2_target):
        task1_loss = self.task1_loss_fn(task1_pred.view(-1), task1_target.float())
        task2_loss = self.task2_loss_fn(task2_pred.view(-1), task2_target.float())

        losses = torch.stack([task1_loss, task2_loss])
        dtype = losses.dtype
        device = losses.device
        self.is_regression = self.is_regression.to(device=device, dtype=dtype)
        std = (torch.exp(self.log_vars)**(1/2)).to(device=device, dtype=dtype)

        coeff = (1/((self.is_regression+1)*(std**2)))
        multi_task_losses = coeff*losses + torch.log(std)

        loss = torch.sum(multi_task_losses)

        return loss, task1_loss, task2_loss
