import numpy as np

from src.my_types import ArrayFloat32Nx2x2


def compute_ftle(flow_map_jacobian: ArrayFloat32Nx2x2, map_period: float):
    # compute the Cauchy-Green deformation tensor
    cauchy_green_tensor = np.einsum(
        "...ji,...jk->...ik", flow_map_jacobian, flow_map_jacobian
    )

    max_eigvals = np.linalg.eigvals(cauchy_green_tensor).max(axis=1)

    return 1 / map_period * np.log(np.sqrt(max_eigvals))
