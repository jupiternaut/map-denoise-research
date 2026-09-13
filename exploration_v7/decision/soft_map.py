"""A pure output action: MAP component, saved soft-M plane, no hard refit."""
from __future__ import annotations
import numpy as np


def project_soft_map(xyz_world_m, order, design_sorted, local_height_sorted_mm,
                     active_sorted, support_sorted, normal_world,
                     group_ids_world, coefficients_mm):
    """Project supported original rows onto their frozen selected soft-M plane.

    Inputs contain measured geometry and legal inference state only. This is
    neither a new association solver nor a posterior-mean output. Inactive and
    unsupported points remain exactly unchanged; labels are not reconsidered.
    """
    world=np.asarray(xyz_world_m,dtype=float)
    order=np.asarray(order,dtype=int)
    design=np.asarray(design_sorted,dtype=float)
    height=np.asarray(local_height_sorted_mm,dtype=float)
    active=np.asarray(active_sorted,dtype=int)
    support=np.asarray(support_sorted,dtype=bool)
    normal=np.asarray(normal_world,dtype=float)
    groups=np.asarray(group_ids_world,dtype=int)
    beta=np.asarray(coefficients_mm,dtype=float)
    n=len(world)
    if world.shape!=(n,3) or not np.isfinite(world).all():raise ValueError('finite world N by 3 required')
    if order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):raise ValueError('order must be a permutation')
    if design.shape!=(n,3) or height.shape!=(n,) or support.shape!=(n,) or groups.shape!=(n,):raise ValueError('state shape mismatch')
    if beta.ndim!=2 or beta.shape[1]!=3 or not np.isfinite(beta).all():raise ValueError('finite plane coefficients required')
    if normal.shape!=(3,) or not np.isclose(np.linalg.norm(normal),1.,atol=1e-12,rtol=0):raise ValueError('unit normal required')
    if len(np.unique(active))!=len(active) or np.any(active<0) or np.any(active>=n):raise ValueError('invalid active rows')
    mask=np.zeros(n,bool);mask[active]=True
    if np.any(support&~mask):raise ValueError('supported row is not an inferred active row')
    labels=groups[order][active]
    if np.any(labels<0) or np.any(labels>=len(beta)):raise ValueError('active row lacks valid component')
    predicted=height.copy()
    predicted[active]=np.sum(design[active]*beta[labels],axis=1)
    ordered=world[order].copy()
    ordered[support]+=(predicted[support]-height[support])[:,None]*normal[None,:]/1000.
    result=world.copy();result[order]=ordered
    world_support=np.empty(n,bool);world_support[order]=support
    if not np.isfinite(result).all():raise ValueError('nonfinite output')
    np.testing.assert_array_equal(result[~world_support],world[~world_support])
    return result,world_support
