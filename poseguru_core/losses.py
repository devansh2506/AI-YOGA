from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


"""
Implements the four PoseGuru loss terms:
- Cpred  : prediction cost (cross-entropy w.r.t target class)
- Cstick : stick length consistency between x* and x'
- Cland  : Procrustes-based landmark alignment to exemplar
- Cangle : angle consistency between x* and exemplar

All tensors use 12 keypoints with 2D coordinates:
index mapping:
0:L_Sh, 1:R_Sh, 2:L_Elb, 3:R_Elb, 4:L_Wr, 5:R_Wr,
6:L_Hip, 7:R_Hip, 8:L_Knee, 9:R_Knee, 10:L_Ank, 11:R_Ank
"""


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:  # (12,2)
        return x.unsqueeze(0)
    return x


def prediction_cost(
    mcls: nn.Module,
    x_star: torch.Tensor,
    target_class: torch.Tensor,
) -> torch.Tensor:
    """
    Cpred: cross-entropy loss between classifier prediction and target class.

    Args:
        mcls: trained classifier taking flattened landmarks (B,24).
        x_star: optimized landmarks (B,12,2) or (12,2).
        target_class: int tensor of shape (B,) or scalar long.
    """
    x_star = _ensure_batch(x_star)  # (B,12,2)
    B = x_star.shape[0]
    x_flat = x_star.view(B, -1)  # (B,24)

    if target_class.dim() == 0:
        target_class = target_class.unsqueeze(0).repeat(B)

    logits = mcls(x_flat)
    loss = F.cross_entropy(logits, target_class)
    return loss


def stick_length_cost(
    x_star: torch.Tensor,
    x_prime: torch.Tensor,
) -> torch.Tensor:
    """
    Cstick: mean absolute deviation of stick lengths between x* and x'.

    Args:
        x_star: (B,12,2)
        x_prime: (B,12,2)
    """
    x_star = _ensure_batch(x_star)
    x_prime = _ensure_batch(x_prime)

    # stick connections in 12-keypoint space
    sticks = [
        # arms
        (0, 2),  # L_Sh -> L_Elb
        (2, 4),  # L_Elb -> L_Wr
        (1, 3),  # R_Sh -> R_Elb
        (3, 5),  # R_Elb -> R_Wr
        # legs
        (6, 8),  # L_Hip -> L_Knee
        (8, 10),  # L_Knee -> L_Ank
        (7, 9),  # R_Hip -> R_Knee
        (9, 11),  # R_Knee -> R_Ank
        # torso
        (0, 1),  # shoulder width
        (6, 7),  # hip width
        (0, 6),  # left torso side
        (1, 7),  # right torso side
    ]

    total = 0.0
    for i, j in sticks:
        curr = torch.norm(x_star[:, i, :] - x_star[:, j, :], dim=-1)
        orig = torch.norm(x_prime[:, i, :] - x_prime[:, j, :], dim=-1)
        total += torch.abs(curr - orig)

    return torch.mean(total)


def _center_and_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Center by centroid and scale-normalize using RMS distance from origin.

    Args:
        x: (B,12,2)
    Returns:
        x_scaled: (B,12,2)
        centroid: (B,1,2)
        scale: (B,1,1)
    """
    centroid = x.mean(dim=1, keepdim=True)  # (B,1,2)
    centered = x - centroid  # (B,12,2)
    # RMS distance from origin
    scale = torch.sqrt((centered ** 2).sum(dim=-1).mean(dim=1, keepdim=True)).unsqueeze(-1)  # (B,1,1)
    # avoid division by zero
    scale = scale + 1e-8
    scaled = centered / scale
    return scaled, centroid, scale


def landmarks_cost(
    x_star: torch.Tensor,
    x_ex: torch.Tensor,
) -> torch.Tensor:
    """
    Cland: L1 distance between centered x* and Procrustes-aligned exemplar.

    Args:
        x_star: (B,12,2)
        x_ex: (B,12,2) exemplar landmarks
    """
    x_star = _ensure_batch(x_star)
    x_ex = _ensure_batch(x_ex)

    # center and scale both
    x_star_scaled, _, _ = _center_and_scale(x_star)  # (B,12,2)
    x_ex_scaled, _, _ = _center_and_scale(x_ex)  # (B,12,2)

    B, N, _ = x_star_scaled.shape

    # reshape to (B,2,N) for cross-covariance
    x_ex_t = x_ex_scaled.permute(0, 2, 1)  # (B,2,12)
    x_star_t = x_star_scaled.permute(0, 2, 1)  # (B,2,12)

    # cross-covariance for each batch item
    H = torch.matmul(x_ex_t, x_star_t.transpose(1, 2))  # (B,2,2)

    # SVD per batch item
    U, S, Vh = torch.linalg.svd(H)
    R = torch.matmul(Vh.transpose(1, 2), U.transpose(1, 2))  # (B,2,2)

    # reflection correction
    det = torch.linalg.det(R)  # (B,)
    mask = det < 0
    if mask.any():
        Vh_corrected = Vh.clone()
        Vh_corrected[mask, -1, :] *= -1
        R = torch.matmul(Vh_corrected.transpose(1, 2), U.transpose(1, 2))

    # rotate exemplar: (B,12,2) = (B,12,2) @ (B,2,2)
    x_ex_rot = torch.matmul(x_ex_scaled, R)  # (B,12,2)

    # L1 distance
    loss = torch.abs(x_ex_rot - x_star_scaled).mean()
    return loss


def get_differentiable_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Calculates the angle at vertex 'b' (a-b-c) in DEGREES, differentiably.

    Args:
        a, b, c: (B,2)
    Returns:
        (B,) tensor of angles in degrees.
    """
    ba = a - b
    bc = c - b
    ba_norm = F.normalize(ba, p=2, dim=-1)
    bc_norm = F.normalize(bc, p=2, dim=-1)
    cosine = (ba_norm * bc_norm).sum(dim=-1)
    cosine = torch.clamp(cosine, -0.999, 0.999)
    return torch.rad2deg(torch.acos(cosine))


def _compute_angles(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the 8 key joint angles from 12-keypoint pose.

    Args:
        points: (B,12,2)
    Returns:
        (B,8) tensor:
        [L_Elbow, R_Elbow, L_Knee, R_Knee, L_Shldr, R_Shldr, L_Hip, R_Hip]
    """
    # elbows
    l_elbow = get_differentiable_angle(points[:, 0], points[:, 2], points[:, 4])
    r_elbow = get_differentiable_angle(points[:, 1], points[:, 3], points[:, 5])
    # knees
    l_knee = get_differentiable_angle(points[:, 6], points[:, 8], points[:, 10])
    r_knee = get_differentiable_angle(points[:, 7], points[:, 9], points[:, 11])
    # shoulders
    l_sh = get_differentiable_angle(points[:, 6], points[:, 0], points[:, 2])
    r_sh = get_differentiable_angle(points[:, 7], points[:, 1], points[:, 3])
    # hips
    l_hip = get_differentiable_angle(points[:, 0], points[:, 6], points[:, 8])
    r_hip = get_differentiable_angle(points[:, 1], points[:, 7], points[:, 9])

    return torch.stack(
        [l_elbow, r_elbow, l_knee, r_knee, l_sh, r_sh, l_hip, r_hip],
        dim=1,
    )


def angle_cost(
    x_star: torch.Tensor,
    x_ex: torch.Tensor,
) -> torch.Tensor:
    """
    Cangle: mean L1 distance between angles of x* and exemplar.

    Args:
        x_star: (B,12,2)
        x_ex: (B,12,2)
    """
    x_star = _ensure_batch(x_star)
    x_ex = _ensure_batch(x_ex)

    angles_star = _compute_angles(x_star)  # (B,8)
    angles_ex = _compute_angles(x_ex)  # (B,8)

    diff = torch.abs(angles_star - angles_ex)
    return diff.mean()


@dataclass
class PoseGuruLossConfig:
    lambda_c: float = 0.1
    lambda_s: float = 0.2
    lambda_l: float = 0.3
    lambda_a: float = 0.4


class PoseGuruLoss(nn.Module):
    """
    Combines the four PoseGuru losses into a single objective.
    """

    def __init__(self, mcls: nn.Module, cfg: PoseGuruLossConfig | None = None):
        super().__init__()
        self.mcls = mcls
        self.cfg = cfg or PoseGuruLossConfig()

    def forward(
        self,
        x_star: torch.Tensor,
        x_prime: torch.Tensor,
        x_ex: torch.Tensor,
        target_class: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x_star: (B,12,2) candidate counterfactual pose
            x_prime: (B,12,2) original incorrect pose
            x_ex: (B,12,2) exemplar pose
            target_class: (B,) long tensor of target class indices
        Returns:
            total_loss, details_dict
        """
        cp = prediction_cost(self.mcls, x_star, target_class)
        cs = stick_length_cost(x_star, x_prime)
        cl = landmarks_cost(x_star, x_ex)
        ca = angle_cost(x_star, x_ex)

        total = (
            self.cfg.lambda_c * cp
            + self.cfg.lambda_s * cs
            + self.cfg.lambda_l * cl
            + self.cfg.lambda_a * ca
        )

        details = {
            "Cpred": cp.detach(),
            "Cstick": cs.detach(),
            "Cland": cl.detach(),
            "Cangle": ca.detach(),
            "total": total.detach(),
        }
        return total, details

