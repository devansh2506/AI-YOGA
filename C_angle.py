import torch
import torch.nn.functional as F

def get_differentiable_angle(a, b, c):
    """
    Calculates the angle at vertex 'b' (a-b-c) differentiably.
    Returns Angle in DEGREES.
    """
    # Vectors: BA and BC
    ba = a - b
    bc = c - b

    # Normalize vectors
    ba_norm = F.normalize(ba, p=2, dim=-1)
    bc_norm = F.normalize(bc, p=2, dim=-1)

    # Dot Product
    cosine_angle = torch.sum(ba_norm * bc_norm, dim=-1)

    # Clamp to avoid NaN
    cosine_angle = torch.clamp(cosine_angle, -0.999, 0.999)

    # Angle in Degrees
    return torch.rad2deg(torch.acos(cosine_angle))

def calculate_angle_loss(user_landmarks, target_angles_list, device='cpu', image_shape=None):
    """
    Calculates the L1 loss between user's current angles and target angles.

    Args:
        user_landmarks: Tensor of shape (Batch, 12, 2).
                        Order MUST be:
                        0:L_Sh, 1:R_Sh, 2:L_Elb, 3:R_Elb, 4:L_Wr, 5:R_Wr,
                        6:L_Hip, 7:R_Hip, 8:L_Knee, 9:R_Knee, 10:L_Ank, 11:R_Ank
        target_angles_list: List of 8 floats [L_Elbow, R_Elbow, L_Knee, R_Knee,
                                              L_Shldr, R_Shldr, L_Hip, R_Hip]
        device: 'cpu' or 'cuda'
        image_shape: Tuple (width, height). Optional. Used to fix aspect ratio.
    """

    # 1. Prepare Target Tensor
    target_tensor = torch.tensor(target_angles_list, dtype=torch.float32, device=device)

    # 2. Aspect Ratio Correction
    # If using normalized [0,1] coords, we must scale to pixels to get correct angles
    if image_shape is not None:
        scale = torch.tensor(image_shape, device=device, dtype=torch.float32).view(1, 1, 2)
        # Scale all 12 points
        points = user_landmarks * scale
    else:
        points = user_landmarks

    # 3. Calculate Angles using the NEW INDICES (0-11)

    # --- ELBOWS ---
    # Left:  L_Sh(0) -> L_Elb(2) -> L_Wr(4)
    l_elbow = get_differentiable_angle(points[:, 0], points[:, 2], points[:, 4])
    # Right: R_Sh(1) -> R_Elb(3) -> R_Wr(5)
    r_elbow = get_differentiable_angle(points[:, 1], points[:, 3], points[:, 5])

    # --- KNEES ---
    # Left:  L_Hip(6) -> L_Knee(8) -> L_Ank(10)
    l_knee = get_differentiable_angle(points[:, 6], points[:, 8], points[:, 10])
    # Right: R_Hip(7) -> R_Knee(9) -> R_Ank(11)
    r_knee = get_differentiable_angle(points[:, 7], points[:, 9], points[:, 11])

    # --- SHOULDERS ---
    # Left:  L_Hip(6) -> L_Sh(0) -> L_Elb(2)
    l_shldr = get_differentiable_angle(points[:, 6], points[:, 0], points[:, 2])
    # Right: R_Hip(7) -> R_Sh(1) -> R_Elb(3)
    r_shldr = get_differentiable_angle(points[:, 7], points[:, 1], points[:, 3])

    # --- HIPS ---
    # Left:  L_Sh(0) -> L_Hip(6) -> L_Knee(8)
    l_hip = get_differentiable_angle(points[:, 0], points[:, 6], points[:, 8])
    # Right: R_Sh(1) -> R_Hip(7) -> R_Knee(9)
    r_hip = get_differentiable_angle(points[:, 1], points[:, 7], points[:, 9])

    # 4. Stack and Calculate Loss
    # Order must match your target_angles_list
    current_angles = torch.stack([
        l_elbow, r_elbow, l_knee, r_knee,
        l_shldr, r_shldr, l_hip, r_hip
    ], dim=1)

    # Calculate L1 distance (Absolute Error)
    diff = torch.abs(current_angles - target_tensor.unsqueeze(0))

    return torch.mean(diff)