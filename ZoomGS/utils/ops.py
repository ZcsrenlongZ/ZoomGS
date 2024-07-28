import numpy as np
import torch
# from utils.typing import *
import torch.nn.functional as F
# from kornia.geometry.depth import depth_to_3d

pytorch3d_capable = True
try:
    import pytorch3d
    from pytorch3d.ops import estimate_pointcloud_normals
    from pytorch3d.ops import sample_farthest_points
    from pytorch3d.ops import knn_points
except ImportError:
    pytorch3d_capable = False


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c




def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()


def angle_bisector(a, b):
    return F.normalize(F.normalize(a, dim=-1) + F.normalize(b, dim=-1), dim=-1)


def estimate_normal(
    pos, neighborhood_size: int = 50, disambiguate_directions: bool = True
):
    if not pytorch3d_capable:
        raise ImportError(
            "pytorch3d is not installed, which is required for normal estimation"
        )

    return estimate_pointcloud_normals(
        pos[None, ...], neighborhood_size, disambiguate_directions
    )[0]


@torch.no_grad()
def farthest_point_sampling(mean: torch.Tensor, K, random_start_point=False):
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for FPS")

    if mean.ndim == 2:
        L = torch.tensor(mean.shape[0], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean[None, ...],
            L[None, ...],
            K,
            random_start_point=random_start_point,
        )
        return pts[0], indices[0]
    elif mean.ndim == 3:
        # mean: [B, L, 3]
        B = mean.shape[0]
        L = torch.tensor(mean.shape[1], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean,
            L[None, ...].repeat(B),
            K,
            random_start_point=random_start_point,
        )

        return pts, indices


@torch.no_grad()
def nearest_neighbor(mean: torch.Tensor):
    if not pytorch3d_capable:
        raise ImportError(
            "pytorch3d is not installed, which is required for nearest neighbor"
        )

    _, idx, nn = knn_points(mean[None, ...], mean[None, ...], K=2, return_nn=True)
    # nn: Tensor of shape (N, P1, K, D)

    # take the index 1 since index 0 is the point itself
    return nn[0, :, 1, :], idx[..., 1][0]


@torch.no_grad()
def K_nearest_neighbors(
    mean: torch.Tensor,
    K: int,
    query= None,
    return_dist=False,
):
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for KNN")
    # TODO: finish this
    if query is None:
        query = mean
    dist, idx, nn = knn_points(query[None, ...], mean[None, ...], K=K, return_nn=True)

    if not return_dist:
        return nn[0, :, 1:, :], idx[0, :, 1:]
    else:
        return nn[0, :, 1:, :], idx[0, :, 1:], dist[0, :, 1:]


def distance_to_gaussian_surface(mean, svec, rotmat, query):
    xyz = query - mean
    # self modified 
    rotmat = qsvec2rotmat_batched(rotmat, svec)
    # TODO: check here
    # breakpoint()
    print(rotmat.transpose(-1, -2).shape, xyz.shape)
    # xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)
    xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)

    xyz = F.normalize(xyz, dim=-1)
    z = xyz[..., 2]
    y = xyz[..., 1]
    x = xyz[..., 0]
    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)
    cos_theta = z
    sin_theta = r_xy
    cos_phi = x / r_xy
    sin_phi = y / r_xy

    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2

    return torch.sqrt(r2 + 1e-10)


def linear_correlation(x, y):
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    return (x * y).sum(dim=-1)



def compute_shaded_color(
    light_pos, light_color, surface_normal, surface_color, mean, cam_pos
):
    ab = angle_bisector(light_pos - mean, cam_pos - mean)
    # backface culling
    dot = (ab * surface_normal).sum(dim=-1).abs().clamp(min=0.0, max=1.0)

    return light_color * dot[..., None] * surface_color


def marching_cubes(density_grid, L, reso, thresh):
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "mcubes is not installed, which is required for marching cubes\nInstall it by `pip install PyMCubes`"
        )
    vertices, triangles = mcubes.marching_cubes(density_grid, thresh)

    return vertices, triangles


import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)
from torchtyping import TensorType


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def qsvec2rotmat_batched(
    qvec: TensorType["N", 4], svec: TensorType["N", 3]
) -> TensorType["N", 3, 3]:
    unscaled_rotmat = quaternion_to_rotation_matrix(qvec, QuaternionCoeffOrder.WXYZ)

    # TODO: check which I current think that scale should be copied row-wise since in eq (6) the S matrix is right-hand multplied to R
    rotmat = svec.unsqueeze(-2) * unscaled_rotmat
    # rotmat = svec.unsqueeze(-1) * unscaled_rotmat
    # rotmat = torch.bmm(unscaled_rotmat, torch.diag(svec))

    # print("rotmat", rotmat.shape)

    return rotmat


def rotmat2wxyz(rotmat):
    return rotation_matrix_to_quaternion(rotmat, order=QuaternionCoeffOrder.WXYZ)


def qvec2rotmat_batched(qvec: TensorType["N", 4]):
    return quaternion_to_rotation_matrix(qvec, QuaternionCoeffOrder.WXYZ)


def qsvec2covmat_batched(qvec: TensorType["N", 4], svec: TensorType["N", 3]):
    rotmat = qsvec2rotmat_batched(qvec, svec)
    return torch.bmm(rotmat, rotmat.transpose(-1, -2))