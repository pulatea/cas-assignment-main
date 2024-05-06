import numpy as np


def pivot_calibration(transforms):
    """
    Pivot calibration

    Keyword arguments:
    transforms -- A list of 4x4 transformation matrices from the tracking system (Fi)
                  representing the tracked tool's position and orientation at
                  different instances.

    Returns:
    T          -- The calibration matrix T (in homogeneous coordinates) that defines
                  the offset (p_t) from the tracked part to the pivot point (tool tip).
    """

    ## TODO: Implement pivot calibration as discussed in the lecture

    # The given4x4 transformation matrices are composed of rotation matrix (3X3), translation vector (3x1)
    # and the last row of homogeneous coordinates.
    # | R t |
    # | 0 1 |

    n = len(transforms)

    R_matrix = np.zeros((n, 3, 3))
    p_vector = np.zeros((n, 3))

    for i, transform in enumerate(transforms):
        R_matrix[i] = transform[:3, :3]  # all rows (except the last one - homogeneous coordinates) until last column
        p_vector[i] = transform[:3, 3]  # all rows (except the last one - homogeneous coordinates) from last column

    # formula for finding pt and pp is R_i * p_t - p_p = - p_i
    # turning the extracted data into a linear system A * x = b, in our case R_matrix * x = p_vector

    # turning rotation matrix into a 2D array
    R_matrix_reshaped = R_matrix.reshape(-1, 3)
    # collapsing translation vector into one dimension
    p_vector_flattened = p_vector.flatten()

    # solving the linear system for x using least squares method
    x = np.linalg.lstsq(R_matrix_reshaped, p_vector_flattened, rcond=None)[0]

    # extracting pt and pp from the vector x
    pt, pp = x[:3], x[3:]

    # Construct the calibration matrix T
    T = np.eye(4)
    T[:3, 3] = pt

    return T


def calibration_device_calibration(camera_T_reference, camera_T_tool, reference_T_tip):
    """
    Tool calibration using calibration device

    Keyword arguments:
    camera_T_reference -- Transformation matrix from reference (calibration device) to camera.
    camera_T_tool      -- Transformation matrix from tool to camera.
    reference_T_tip    -- Transformation matrix from tip to reference (calibration device).

    Returns:
    T                  -- Calibration matrix from tool to tip.
    """

    ## TODO: Implement a calibration method which uses a calibration device

    T = np.eye(4)

    # from the lecture we have:
    # tool_T_tip = tool_T_camera * camera_T_ref * ref_T_tip
    # tool_T_tip = (camera_T_tool)_inverted * camera_T_ref * ref_T_tip

    camera_T_tool_inverted = np.linalg.inv(camera_T_tool)

    T = camera_T_tool_inverted @ camera_T_reference @ reference_T_tip

    return T
