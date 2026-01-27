import torch



def calculate_stick_loss(current_landmarks, original_landmarks):
    """
    Computes the L1 difference between bone lengths in the current optimizing pose
    and the original user pose.

    Args:
        current_landmarks: Tensor (Batch, 33, 2). The optimizing pose (requires_grad=True).
        original_landmarks: Tensor (Batch, 33, 2). The fixed original pose (No grads).

    Returns:
        loss: Scalar tensor.
    """

    # 1. Define the specific connections (Sticks) based on your LANDMARK_MAP
    # L_Sh=11, R_Sh=12, L_Elb=13, R_Elb=14, L_Wr=15, R_Wr=16
    # L_Hip=23, R_Hip=24, L_Knee=25, R_Knee=26, L_Ank=27, R_Ank=28

    sticks = [
        # --- ARMS ---
        (11, 13),  # Left Shoulder -> Left Elbow
        (13, 15),  # Left Elbow -> Left Wrist
        (12, 14),  # Right Shoulder -> Right Elbow
        (14, 16),  # Right Elbow -> Right Wrist

        # --- LEGS ---
        (23, 25),  # Left Hip -> Left Knee
        (25, 27),  # Left Knee -> Left Ankle
        (24, 26),  # Right Hip -> Right Knee
        (26, 28),  # Right Knee -> Right Ankle

        # --- TORSO STRUCTURE (Optional but recommended) ---
        (11, 12),  # Shoulder Width
        (23, 24),  # Hip Width
        (11, 23),  # Left Torso Side
        (12, 24)  # Right Torso Side
    ]

    total_loss = 0.0

    # 2. Loop through every stick connection
    for start_idx, end_idx in sticks:
        # A. Get vectors for the Current Pose (Optimizing)
        curr_p1 = current_landmarks[:, start_idx, :]
        curr_p2 = current_landmarks[:, end_idx, :]

        # B. Get vectors for the Original Pose (Fixed Target)
        orig_p1 = original_landmarks[:, start_idx, :]
        orig_p2 = original_landmarks[:, end_idx, :]

        # C. Calculate Lengths (Euclidean Distance)
        # norm(p1 - p2) is equivalent to sqrt((x1-x2)^2 + (y1-y2)^2)
        curr_dist = torch.norm(curr_p1 - curr_p2, dim=-1)
        orig_dist = torch.norm(orig_p1 - orig_p2, dim=-1)

        # D. Calculate Difference (L1 Loss)
        # We want the current length to match the original length exactly
        total_loss += torch.abs(curr_dist - orig_dist)

    # 3. Return the average loss
    return torch.mean(total_loss)