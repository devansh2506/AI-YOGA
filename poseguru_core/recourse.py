from __future__ import annotations

from typing import Tuple

import torch

from poseguru_core.losses import PoseGuruLoss


def run_recourse(
    x_prime: torch.Tensor,
    x_ex: torch.Tensor,
    target_class: torch.Tensor,
    loss_module: PoseGuruLoss,
    num_steps: int = 200,
    lr: float = 5e-2,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Runs gradient-based exemplar-driven algorithmic recourse to obtain x* from x'.

    Args:
        x_prime: (B,12,2) incorrect pose landmarks.
        x_ex: (B,12,2) exemplar landmarks for the target class.
        target_class: (B,) long tensor of target labels.
        loss_module: PoseGuruLoss instance (wraps Mcls and lambda weights).
        num_steps: number of optimization iterations.
        lr: learning rate for optimizer (NAdam).
        device: optional device override.

    Returns:
        x_star: optimized landmarks (B,12,2)
        log: dict with loss history (every step).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_prime = x_prime.to(device)
    x_ex = x_ex.to(device)
    target_class = target_class.to(device)
    loss_module = loss_module.to(device)

    # initialize x* from x'
    x_star = x_prime.clone().detach().requires_grad_(True)

    optimizer = torch.optim.NAdam([x_star], lr=lr)

    history = {
        "total": [],
        "Cpred": [],
        "Cstick": [],
        "Cland": [],
        "Cangle": [],
    }

    for step in range(num_steps):
        optimizer.zero_grad()
        total_loss, details = loss_module(x_star, x_prime, x_ex, target_class)
        total_loss.backward()
        optimizer.step()

        history["total"].append(float(details["total"].cpu()))
        history["Cpred"].append(float(details["Cpred"].cpu()))
        history["Cstick"].append(float(details["Cstick"].cpu()))
        history["Cland"].append(float(details["Cland"].cpu()))
        history["Cangle"].append(float(details["Cangle"].cpu()))

    return x_star.detach().cpu(), history

