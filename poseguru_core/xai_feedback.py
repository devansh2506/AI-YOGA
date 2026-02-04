import numpy as np
from typing import List, Tuple

JOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def generate_feedback(current_pose: np.ndarray, corrected_pose: np.ndarray, threshold: float = 0.20) -> List[str]:
    """
    Generates text feedback based on the difference between current and corrected pose.
    
    Args:
        current_pose: (12, 2) array of normalized (x, y) coordinates
        corrected_pose: (12, 2) array of normalized (x, y) coordinates
        threshold: minimum valid difference to trigger feedback
        
    Returns:
        List of strings with feedback instructions
    """
    # Vector diff: corrected - current
    diff = corrected_pose - current_pose
    
    # Calculate magnitude of deviation for each joint
    magnitudes = np.linalg.norm(diff, axis=1)
    
    # Find joints with significant deviation
    significant_indices = np.where(magnitudes > threshold)[0]
    
    if len(significant_indices) == 0:
        return ["Great pose! Hold it."]
        
    # Sort by deviation magnitude (descending) to prioritize worst errors
    sorted_indices = significant_indices[np.argsort(-magnitudes[significant_indices])]
    
    feedback_msgs = []
    
    # Limit to top 2 corrections to avoid overwhelming the user
    for idx in sorted_indices[:2]:
        joint_name = JOINT_NAMES[idx].replace("_", " ")
        dx, dy = diff[idx]
        
        # Determine direction
        horizontal_msg = ""
        vertical_msg = ""
        
        # Note: Y-axis is inverted in image coordinates (0 is top)
        # So negative dy means "move up" (towards 0)
        
        if abs(dx) > abs(dy):
            # Primary movement is horizontal
            if dx > 0:
                horizontal_msg = "right"
            else:
                horizontal_msg = "left"
            msg = f"Move your {joint_name} to the {horizontal_msg}"
        else:
            # Primary movement is vertical
            if dy < 0:
                vertical_msg = "up"
            else:
                vertical_msg = "down"
            msg = f"Move your {joint_name} {vertical_msg}"
            
        feedback_msgs.append(msg)
        
    return feedback_msgs
