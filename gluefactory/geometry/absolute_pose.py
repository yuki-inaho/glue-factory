import kornia
import torch
from kornia.geometry.calibration.pnp import _mean_isotropic_scale_normalize
from torch import Tensor, arange

from . import transforms as gtr
from .reconstruction import Pose


def pnp_dlt(p3d_w: Tensor, p2d_c: Tensor) -> Pose:
    # p3d_w: (B, N, 3) - 3D points in world coordinates
    # p2d_c: (B, N, 2) - 2D points in camera coordinates (normalized)
    B, N = p3d_w.shape[:2]

    p3d_w_norm, world_transform_norm = _mean_isotropic_scale_normalize(p3d_w)
    p3d_w_norm_h = gtr.to_homogeneous(p3d_w_norm)
    img_points_norm, img_transform_norm = _mean_isotropic_scale_normalize(p2d_c)
    inv_img_transform_norm = torch.inverse(img_transform_norm)

    # Setting up the system (the matrix A in Ax=0)
    system = torch.zeros((B, 2 * N, 12), dtype=p3d_w.dtype, device=p3d_w.device)
    system[:, 0::2, 0:4] = p3d_w_norm_h
    system[:, 1::2, 4:8] = p3d_w_norm_h
    system[:, 0::2, 8:12] = p3d_w_norm_h * (-1) * img_points_norm[..., 0:1]
    system[:, 1::2, 8:12] = p3d_w_norm_h * (-1) * img_points_norm[..., 1:2]

    # Getting the solution vectors.
    _, _, v = torch.svd(system)
    solution = v[..., -1]

    # Reshaping the solution vectors to the correct shape.
    solution = solution.reshape(B, 3, 4)

    # Creating solution_4x4
    solution_4x4 = kornia.utils.eye_like(4, solution)
    solution_4x4[:, :3, :] = solution

    # De-normalizing the solution
    intermediate = torch.bmm(solution_4x4, world_transform_norm)
    solution = torch.bmm(inv_img_transform_norm, intermediate[:, :3, :])

    # We obtained one solution for each element of the batch. We may
    # need to multiply each solution with a scalar. This is because
    # if x is a solution to Ax=0, then cx is also a solution. We can
    # find the required scalars by using the properties of
    # rotation matrices. We do this in two parts:

    # First, we fix the sign by making sure that the determinant of
    # the all the rotation matrices are non negative (since determinant
    # of a rotation matrix should be 1).
    det = torch.det(solution[:, :3, :3])
    ones = torch.ones_like(det)
    sign_fix = torch.where(det < 0, ones * -1, ones)
    solution = solution * sign_fix[:, None, None]

    # Then, we make sure that norm of the 0th columns of the rotation
    # matrices are 1. Do note that the norm of any column of a rotation
    # matrix should be 1. Here we use the 0th column to calculate norm_col.
    # We then multiply solution with mul_factor.
    norm_col = torch.norm(input=solution[:, :3, 0], p=2, dim=1)
    mul_factor = (1 / norm_col)[:, None, None]
    temp = solution * mul_factor

    # To make sure that the rotation matrix would be orthogonal, we apply
    # QR decomposition.
    ortho, right = torch.linalg.qr(temp[:, :3, :3])

    # We may need to fix the signs of the columns of the ortho matrix.
    # If right[i, j, j] is negative, then we need to flip the signs of
    # the column ortho[i, :, j]. The below code performs the necessary
    # operations in an better way.
    mask = kornia.utils.eye_like(3, ortho)
    col_sign_fix = torch.sign(mask * right)
    rot_mat = torch.bmm(ortho, col_sign_fix)

    return Pose.from_Rt(rot_mat, temp[:, :3, 3])
