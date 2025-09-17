import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Define the expanded feature names including engineered features
FEATURE_NAMES = [
    # Original features
    "acc_x", "acc_y", "acc_z",
    "rot_6d_0", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5", "angular_vel_x", "angular_vel_y", "angular_vel_z",
    "linear_acc_x", "linear_acc_y", "linear_acc_z",
]

def normalize_quaternion(quat):
    """Normalize quaternion to unit length"""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)  # Avoid division by zero
    return quat / norm

def quaternion_to_6d_rotation(quat):
    """Convert quaternion to 6D rotation representation
    
    Args:
        quat: quaternion array with shape (..., 4) in [x, y, z, w] format
        
    Returns:
        6D rotation representation with shape (..., 6)
    """
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)
    
    # Handle NaN values
    has_nan = np.any(np.isnan(quat), axis=-1)
    result = np.full((*quat.shape[:-1], 6), np.nan)
    
    # Process only valid quaternions
    valid_mask = ~has_nan & ~np.all(np.isclose(quat, 0), axis=-1)
    if not np.any(valid_mask):
        return result
    
    valid_quat = quat[valid_mask]
    
    try:
        # Normalize quaternions
        valid_quat_norm = normalize_quaternion(valid_quat)
        
        # Convert to rotation matrices
        rotations = R.from_quat(valid_quat_norm)
        rotation_matrices = rotations.as_matrix()
        
        # Extract first two columns as 6D representation
        # This gives us two orthonormal 3D vectors
        result[valid_mask] = rotation_matrices[:, :, :2].reshape(-1, 6)
        
    except (ValueError, RuntimeError):
        # If conversion fails, keep as NaN
        pass
    
    return result

def remove_gravity_from_acc(acc_data, rot_data, gravity_world=np.array([0, 0, 9.81])):
    """Remove gravity component from acceleration data"""
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.full_like(acc_values, np.nan)  # Initialize with NaN
    
    for i in range(num_samples):
        # If either acc or rot data is NaN, keep result as NaN
        if np.any(np.isnan(acc_values[i])) or np.any(np.isnan(quat_values[i])):
            linear_accel[i, :] = np.nan
            continue
            
        if np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            # Normalize quaternion
            quat_norm = normalize_quaternion(quat_values[i:i+1])[0]
            rotation = R.from_quat(quat_norm)
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, RuntimeError):
            linear_accel[i, :] = np.nan  # Set to NaN if computation fails
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    """Calculate angular velocity from quaternion sequence"""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.full((num_samples, 3), np.nan)  # Initialize with NaN

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        # If either quaternion has NaN, set result to NaN
        if np.any(np.isnan(q_t)) or np.any(np.isnan(q_t_plus_dt)) or \
           np.all(np.isclose(q_t, 0)) or np.all(np.isclose(q_t_plus_dt, 0)):
            angular_vel[i, :] = np.nan
            continue

        try:
            # Normalize quaternions
            q_t_norm = normalize_quaternion(q_t.reshape(1, -1))[0]
            q_t_plus_dt_norm = normalize_quaternion(q_t_plus_dt.reshape(1, -1))[0]
            
            rot_t = R.from_quat(q_t_norm)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt_norm)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, RuntimeError):
            # If quaternion is invalid, set to NaN
            angular_vel[i, :] = np.nan
            
    return angular_vel