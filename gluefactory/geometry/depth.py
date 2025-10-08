import functools

import kornia
import torch

from ..utils import misc
from . import reconstruction


def shape_normalize(kpts, w, h):
    """Normalize points to [-1, 1] range."""
    kpts = kpts.clone()
    kpts[..., 0] = kpts[..., 0] * 2 / w - 1
    kpts[..., 1] = kpts[..., 1] * 2 / h - 1

    kpts = kpts[:, None]
    return kpts


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = shape_normalize(pts, w, h)
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, torch.nan)
    if depth_.dim() == 2:
        depth = depth[None]
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    if depth_.dim() == 2:
        interp = interp[0]
        valid = valid[0]
    interp[~valid] = 0.0
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj
    if depthj is None or ccth is None:
        return kpi_j, validi & validj
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
        consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
        visible = validi & consistent & validj_i & validj
        # visible = validi
        return kpi_j, visible


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: reconstruction.Camera,
    cameraj: reconstruction.Camera,
    **kwargs,
):
    kpi = misc.get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthi.shape[-2:])
    )


def symmetric_reprojection_error(
    pts0: torch.Tensor,  # B x N x 2
    pts1: torch.Tensor,  # B x N x 2
    camera0: reconstruction.Camera,
    camera1: reconstruction.Camera,
    T_0to1: reconstruction.Pose,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    T_1to0 = T_0to1.inv()
    d0, valid0 = sample_depth(pts0, depth0)
    d1, valid1 = sample_depth(pts1, depth1)

    pts0_1, visible0 = project(
        pts0, d0, depth1, camera0, camera1, T_0to1, valid0, ccth=None
    )
    pts1_0, visible1 = project(
        pts1, d1, depth0, camera1, camera0, T_1to0, valid1, ccth=None
    )

    reprojection_errors_px = 0.5 * (
        (pts0_1 - pts1).norm(dim=-1) + (pts1_0 - pts0).norm(dim=-1)
    )

    valid = valid0 & valid1
    return reprojection_errors_px, valid


def align_pointclouds(
    pts_v0: torch.Tensor,
    pts_v1: torch.Tensor,
    weights: torch.Tensor = None,
    return_Rt: bool = False,
) -> tuple[
    reconstruction.Pose | None | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Estimate a similarity transformation (sim3) between two point clouds."""
    assert pts_v0.shape == pts_v1.shape, f"{pts_v0.shape} != {pts_v1.shape}"
    assert pts_v0.shape[-1] == 3 and len(pts_v0.shape) == 2, f"{pts_v0.shape}"

    pts_v1_in = pts_v1.clone()
    # estimate a sim3 transformation to align two point clouds
    # find M = argmin ||P1 - M @ P2||
    if weights is None:
        weights = torch.ones_like(pts_v0[..., 0])
    weights = weights[:, None]

    t0 = misc.wmean(pts_v0, weights, dim=0)
    t1 = misc.wmean(pts_v1, weights, dim=0)
    pts_v0 = pts_v0 - t0[None, :]
    pts_v1 = pts_v1 - t1[None, :]

    s0 = misc.wmean(pts_v0.square().sum(dim=-1), weights[:, 0]).sqrt()
    s1 = misc.wmean(pts_v1.square().sum(dim=-1), weights[:, 0]).sqrt()

    pts_v0 = pts_v0 / s0
    pts_v1 = pts_v1 / s1

    pts_v0 = pts_v0 * weights
    # Do not mult here as this is used in the output
    # pts_v1 = pts_v1 * weights
    try:
        U, _, V = (pts_v0.T @ pts_v1).double().svd()
        U: torch.Tensor = U
        V: torch.Tensor = V
    except:
        print("Procustes failed: SVD did not converge!")
        s = s0 / s1
        return None, s, pts_v1
    # build rotation matrix
    R = (U @ V.T).float()
    R = torch.stack(
        [R[:, 0], R[:, 1], R[:, 2] * R.det().sign()], dim=-1
    )  # ensure a right-handed coordinate system
    s = s0 / s1
    t = t0 - s * (t1 @ R.T)
    c0_t_c1 = reconstruction.Pose.from_Rt(R, t)
    pts1_v0 = c0_t_c1.transform(pts_v1_in * s)
    if return_Rt:
        return (R, t), s, pts1_v0
    else:
        return c0_t_c1, s, pts1_v0


def batch_align_pointclouds(
    pts_v0: torch.Tensor,
    pts_v1: torch.Tensor,
    weights: torch.Tensor = None,
) -> tuple[reconstruction.Pose | None, torch.Tensor, torch.Tensor]:

    in_dims = (0, 0, 0) if weights is not None else (0, 0)

    c0_Rt_c1, scales, pts1_v0 = torch.vmap(
        functools.partial(align_pointclouds, return_Rt=True),
        in_dims=in_dims,
        out_dims=0,
    )(pts_v0, pts_v1, weights)

    return reconstruction.Pose.from_Rt(*c0_Rt_c1), scales, pts1_v0
