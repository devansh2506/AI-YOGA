from __future__ import annotations

from typing import Tuple

import torch


"""
Implements the pose refinement step (Algorithm 1 in the PoseGuru paper)
and the corresponding action vector.

Assumes 12-keypoint indexing:
0:L_Sh, 1:R_Sh, 2:L_Elb, 3:R_Elb, 4:L_Wr, 5:R_Wr,
6:L_Hip, 7:R_Hip, 8:L_Knee, 9:R_Knee, 10:L_Ank, 11:R_Ank
"""


SEGMENTS = {
    "la": [0, 2, 4],  # left arm
    "ra": [1, 3, 5],  # right arm
    "ll": [6, 8, 10],  # left leg
    "rl": [7, 9, 11],  # right leg
    "t": [0, 1, 6, 7],  # torso
}


def refine_pose(
    x_prime: torch.Tensor,
    x_star: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the PoseGuru refinement:
    - compute L1 deviation for each body segment between x* and x'
    - keep only the segment with maximum deviation from x*, reset others to x'

    Args:
        x_prime: (B,12,2) original incorrect pose
        x_star: (B,12,2) optimized pose

    Returns:
        x_final: (B,12,2) refined corrected pose
        deltas: (B,12,2) action vector (x_final - x_prime)
    """
    if x_prime.dim() == 2:
        x_prime = x_prime.unsqueeze(0)
    if x_star.dim() == 2:
        x_star = x_star.unsqueeze(0)

    B = x_prime.shape[0]
    x_final = x_prime.clone()

    # compute deviations per segment
    seg_devs = []
    for key, indices in SEGMENTS.items():
        idx = torch.tensor(indices, dtype=torch.long, device=x_prime.device)
        diff = torch.abs(x_star[:, idx, :] - x_prime[:, idx, :])
        dev = diff.sum(dim=[1, 2])  # (B,)
        seg_devs.append(dev.unsqueeze(1))

    seg_devs_tensor = torch.cat(seg_devs, dim=1)  # (B,5)
    # segment order is [la, ra, ll, rl, t]
    max_idx = seg_devs_tensor.argmax(dim=1)  # (B,)

    seg_keys = ["la", "ra", "ll", "rl", "t"]

    # For each batch item, keep only the max-deviation segment from x_star
    for b in range(B):
        keep_key = seg_keys[int(max_idx[b])]
        keep_indices = torch.tensor(SEGMENTS[keep_key], dtype=torch.long, device=x_prime.device)
        x_final[b, keep_indices, :] = x_star[b, keep_indices, :]

    deltas = x_final - x_prime
    return x_final, deltas

