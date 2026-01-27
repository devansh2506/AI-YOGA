from __future__ import annotations

from typing import Tuple

import torch


def compute_mpi_jad_and_pcik(
    x_gt: torch.Tensor,
    x_prime: torch.Tensor,
    x_final: torch.Tensor,
    threshold: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Compute MPIJAD and PCIK metrics aggregated over a batch of samples.

    Args:
        x_gt: (B,12,2) ground-truth landmarks.
        x_prime: (B,12,2) incorrect landmarks.
        x_final: (B,12,2) corrected landmarks.
        threshold: distance threshold in normalized coordinates.

    Returns:
        mpi_jad: mean per incorrect joint absolute deviation after correction.
        pcp_mpi_jad: fraction of poses whose mean incorrect-joint deviation <= threshold.
        pcik: percentage of originally incorrect joints that become correct after correction.
    """
    if x_gt.dim() == 2:
        x_gt = x_gt.unsqueeze(0)
    if x_prime.dim() == 2:
        x_prime = x_prime.unsqueeze(0)
    if x_final.dim() == 2:
        x_final = x_final.unsqueeze(0)

    B, J, _ = x_gt.shape

    # initial and final joint errors
    init_err = torch.norm(x_prime - x_gt, dim=-1)  # (B,J)
    final_err = torch.norm(x_final - x_gt, dim=-1)  # (B,J)

    incorrect_mask = init_err > threshold  # (B,J)

    # avoid division by zero for samples with no incorrect joints
    num_incorrect_per_sample = incorrect_mask.sum(dim=1)  # (B,)
    valid_samples_mask = num_incorrect_per_sample > 0

    if not valid_samples_mask.any():
        return 0.0, 0.0, 0.0

    # MPIJAD: mean per incorrect joint absolute deviation after correction
    masked_final_err = final_err[incorrect_mask]  # (N_incorrect,)
    mpi_jad = float(masked_final_err.mean().item())

    # PCP@MPIJAD: fraction of poses where mean incorrect-joint deviation <= threshold
    per_sample_incorrect_means = []
    for b in range(B):
        if not valid_samples_mask[b]:
            continue
        errs_b = final_err[b][incorrect_mask[b]]
        per_sample_incorrect_means.append(errs_b.mean())

    if per_sample_incorrect_means:
        per_sample_means_tensor = torch.stack(per_sample_incorrect_means)
        pcp_mpi_jad = float((per_sample_means_tensor <= threshold).float().mean().item())
    else:
        pcp_mpi_jad = 0.0

    # PCIK: percentage of originally incorrect joints that become correct
    corrected_mask = (init_err > threshold) & (final_err <= threshold)
    num_corrected = corrected_mask.sum().item()
    num_initial_incorrect = incorrect_mask.sum().item()
    pcik = float(num_corrected / num_initial_incorrect) if num_initial_incorrect > 0 else 0.0

    return mpi_jad, pcp_mpi_jad, pcik

