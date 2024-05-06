import numpy as np


def paired_point_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target point clouds.
    :param source: A N x 3 matrix with N 3D points.
    :param target: A N x 3 matrix with N 3D points.
    :return:
        T: 4x4 transformation matrix mapping source to target.
        R: 3x3 rotation matrix part of T.
        t: 1x3 translation vector part of T.
    """
    assert source.shape == target.shape
    T = np.eye(4)
    R = np.eye(3)
    t = np.zeros((1, 3))

    ## TODO: your code goes here
    # source = floating dataset; target = reference dataset, or the opposite.
    # for each point in floating dataset find the corresponding point in the reference dataset p_r = R * p_l + t
    # first, all points should be referred to a coordinate system relative to the centroid
    source_mu = np.mean(source, axis=0)
    target_mu = np.mean(target, axis=0)

    # point clouds should be centered using the centroids
    source_centered = source - source_mu
    target_centered = target - target_mu

    # constructing the covariance matrix
    cov_matrix_M = np.dot(source_centered.T, target_centered)

    # singular value decomposition (SVD) of an 𝑚 by 𝑚 matrix 𝑀 --> M = U * W * V.T
    U, W, V_T = np.linalg.svd(cov_matrix_M)

    # the orthonormal matrix (rotation matrix) 𝑅 that maximizes the trace 𝑡𝑟 𝑅 ⋅ 𝑀 is: 𝑅 = 𝑉 ⋅ 𝑈⊤
    R = np.dot(V_T.T, U.T)

    # translation vector 𝑡 = 𝜇𝑟 − 𝑅 𝜇𝑙
    t = target_mu - np.dot(R, source_mu)

    # initializing the transformation matrix it's rotation and translation properties
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t


def get_initial_pose(source, target):
    """
    Calculates an initial rough registration or optionally returns a hand-picked initial pose.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: An initial 4 x 4 rigid transformation matrix.
    """
    T = np.eye(4)

    ## TODO: Your code goes here
    # when printing the shapes of source and target matrices we see that their shapes are non-matching
    # we have to align the dimensions using PCA to be able to perform decomposition later
    source_mu = np.mean(source, axis=0)
    target_mu = np.mean(target, axis=0)

    source_pca = PCA(n_components=3)
    target_pca = PCA(n_components=3)
    source_pca.fit(source - source_mu)
    target_pca.fit(target - target_mu)

    R = np.dot(target_pca.components_.T, source_pca.components_)

    t = target_mu - np.dot(R, source_mu)

    T[:3, :3] = R
    T[:3, 3] = t

    return T


def find_nearest_neighbor(source, target):
    """
    Finds the nearest neighbor in 'target' for every point in 'source'.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :return: A tuple containing two arrays: the first array contains the
             distances to the nearest neighbor in 'target' for each point
             in 'source', and the second array contains the indices of
             these nearest neighbors in 'target'.
    """

    # build tree containing on target points
    k_d_tree = KDTree(target)

    # find nearest neighbor point using source points
    neighbors, ids = k_d_tree.query(source, k=1)

    return neighbors, ids


def icp(source, target, init_pose=None, max_iterations=10, tolerance=0.0001):
    """
    Iteratively finds the best transformation mapping the source points onto the target.
    :param source: A N x 3 point cloud.
    :param target: A N x 3 point cloud.
    :param init_pose: Initial pose as a 4 x 4 transformation matrix.
    :param max_iterations: Maximum iterations.
    :param tolerance: Error tolerance.
    :return: The optimal 4 x 4 rigid transformation matrix, distances, and registration error.
    """

    # Initialisation
    T = np.eye(4)
    distances = 0
    error = np.finfo(float).max

    ## TODO: Your code goes here

    T = init_pose

    for i in range(max_iterations):
        # for each point in the source point cloud match the closest point in the reference point cloud
        nearest_neighbors, nearest_ids = find_nearest_neighbor(source, target)
        T, R, t = paired_point_matching(source, target[nearest_ids])
        source = (T @ np.hstack([source, np.ones((source.shape[0], 1))]).T).T[:, :3]
        error = np.mean(nearest_neighbors)
        if np.abs(error - distances) < tolerance:
            break
        distances = error

    return T, distances, error
