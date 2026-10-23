"""Controlled response features and a fixed, null-trained added-information test.

These are benchmark summaries, not a new physical model or a fractal dimension.
"""
from __future__ import annotations
import math
import numpy as np
from .multiscale import checked_patches, multiscale_reference, multiscale_cuda

EULER_LEVELS = (-1., 0., 1.)
RIDGE = .1
SHRINKAGE = .1


def spherical_patch_locations(theta, phi, angles, size=64, width_degrees=20.):
    """Gnomonic grids at rotations about one fixed spherical patch-center axis."""
    angles = np.asarray(angles, dtype=np.float64)
    if (not np.isfinite(theta) or not 0 <= theta <= np.pi or not np.isfinite(phi)
            or angles.ndim != 1 or not len(angles) or not np.all(np.isfinite(angles))
            or type(size) is not int or size < 2 or not np.isfinite(width_degrees)
            or not 0 < width_degrees < 90):
        raise ValueError('invalid spherical grid specification')
    center = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    east = np.array([-np.sin(phi), np.cos(phi), 0.])
    north = np.array([-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)])
    coordinate = np.linspace(-np.tan(np.radians(width_degrees/2)), np.tan(np.radians(width_degrees/2)), size)
    xx, yy = np.meshgrid(coordinate, coordinate)
    locations = []
    for angle in angles:
        right = np.cos(angle)*east + np.sin(angle)*north
        up = -np.sin(angle)*east + np.cos(angle)*north
        vectors = center + xx[..., None]*right + yy[..., None]*up
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        locations.append(np.stack([np.arccos(np.clip(vectors[..., 2], -1., 1.)),
            np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), 2*np.pi)], axis=-1))
    return np.ascontiguousarray(locations)


def excursion_euler_reference(patches):
    """V-E+F of upper-excursion vertex cubical complexes, divided by grid size.

    An edge requires both endpoints; a filled face requires all four vertices.
    These Euler characteristics are established geometric summaries, not PD0.
    """
    x = checked_patches(patches)
    std = x.std(axis=(-2, -1), keepdims=True)
    if np.any(std <= 0):
        raise ValueError('constant patch')
    z = (x-x.mean(axis=(-2, -1), keepdims=True))/std
    values = []
    for threshold in EULER_LEVELS:
        a = z > threshold
        vertices = a.sum(axis=(-2, -1))
        edges = (a[:, 1:, :]&a[:, :-1, :]).sum(axis=(-2, -1)) + (a[:, :, 1:]&a[:, :, :-1]).sum(axis=(-2, -1))
        faces = (a[:, :-1, :-1]&a[:, 1:, :-1]&a[:, :-1, 1:]&a[:, 1:, 1:]).sum(axis=(-2, -1))
        values.append((vertices-edges+faces)/(x.shape[-2]*x.shape[-1]))
    return np.stack(values, axis=1)


def checked_response_fields(fields):
    x = np.asarray(fields, dtype=np.float64)
    if (x.ndim != 6 or not len(x) or x.shape[1:4] != (3, 4, 2)
            or min(x.shape[-2:]) <= 16 or not np.all(np.isfinite(x))):
        raise ValueError('need finite [batch,3 smoothings,4 angles,2 representations,height,width]')
    return x


def extract_response_reference(fields):
    """17 conventional baseline values and 90 fixed response contrasts per sky."""
    x = checked_response_fields(fields)
    flat = x.reshape(-1, *x.shape[-2:])
    curves, feature = multiscale_reference(flat)
    geometry = np.concatenate([feature, excursion_euler_reference(flat)], axis=1).reshape(len(x), 3, 4, 2, 9)
    y = x[:, 0, 0, 0]
    std = y.std(axis=(-2, -1))
    z = (y-y.mean(axis=(-2, -1), keepdims=True))/std[:, None, None]
    moments = np.column_stack([np.log(std), np.mean(z**3, axis=(-2, -1)), np.mean(z**4, axis=(-2, -1))-3])
    base_curves = curves.reshape(len(x), 3, 4, 2, 5, 6)[:, 0, 0, 0]
    baseline = np.concatenate([moments, np.log(base_curves[:, :, 0]/std[:, None]**2), geometry[:, 0, 0, 0]], axis=1)
    orientation = geometry[:, :, 1:].mean(axis=2)-geometry[:, :, 0]
    smoothing = geometry[:, 1:, 0]-geometry[:, :1, 0]
    response = np.concatenate([orientation.reshape(len(x), -1), smoothing.reshape(len(x), -1)], axis=1)
    return baseline, response


def extract_response_cuda(fields, device='cuda:0'):
    """Float64 CUDA moments, excursion complexes and response aggregation."""
    import torch
    x = checked_response_fields(fields)
    flat = x.reshape(-1, *x.shape[-2:])
    curves, feature = multiscale_cuda(flat, device=device)
    t = torch.as_tensor(x, dtype=torch.float64, device=device)
    image = t.reshape(-1, *x.shape[-2:])
    deviation = image.std(dim=(-2, -1), correction=0, keepdim=True)
    if bool(torch.any(deviation <= 0).item()):
        raise ValueError('constant patch')
    z = (image-image.mean(dim=(-2, -1), keepdim=True))/deviation
    euler = []
    for threshold in EULER_LEVELS:
        a = z > threshold
        vertices = a.sum(dim=(-2, -1))
        edges = (a[:, 1:, :]&a[:, :-1, :]).sum(dim=(-2, -1)) + (a[:, :, 1:]&a[:, :, :-1]).sum(dim=(-2, -1))
        faces = (a[:, :-1, :-1]&a[:, 1:, :-1]&a[:, :-1, 1:]&a[:, 1:, 1:]).sum(dim=(-2, -1))
        euler.append((vertices-edges+faces).to(torch.float64)/(x.shape[-2]*x.shape[-1]))
    geometry = torch.cat([torch.as_tensor(feature, dtype=torch.float64, device=device),
        torch.stack(euler, dim=1)], dim=1).reshape(len(x), 3, 4, 2, 9)
    y = t[:, 0, 0, 0]
    std = y.std(dim=(-2, -1), correction=0)
    zy = (y-y.mean(dim=(-2, -1), keepdim=True))/std[:, None, None]
    moments = torch.stack([torch.log(std), (zy**3).mean(dim=(-2, -1)), (zy**4).mean(dim=(-2, -1))-3], dim=1)
    c = torch.as_tensor(curves, dtype=torch.float64, device=device).reshape(len(x), 3, 4, 2, 5, 6)[:, 0, 0, 0]
    baseline = torch.cat([moments, torch.log(c[:, :, 0]/std[:, None]**2), geometry[:, 0, 0, 0]], dim=1)
    orientation = geometry[:, :, 1:].mean(dim=2)-geometry[:, :, 0]
    smoothing = geometry[:, 1:, 0]-geometry[:, :1, 0]
    response = torch.cat([orientation.reshape(len(x), -1), smoothing.reshape(len(x), -1)], dim=1)
    return baseline.cpu().numpy(), response.cpu().numpy()


def checked_feature_pair(baseline, response):
    b, r = np.asarray(baseline, dtype=np.float64), np.asarray(response, dtype=np.float64)
    if b.ndim != 2 or r.ndim != 2 or not len(b) or len(b) != len(r) or not b.shape[1] or not r.shape[1] or not np.all(np.isfinite(b)) or not np.all(np.isfinite(r)):
        raise ValueError('need matching finite feature matrices')
    return b, r


def fit_response_model(baseline, response):
    """Fixed ridge residualization and shrinkage Mahalanobis anomaly scores.

    Fit only null training draws. Residualization is approximate; it does not
    guarantee nuisance immunity, independence, or added scientific information.
    """
    b, r = checked_feature_pair(baseline, response)
    if len(b) < 3:
        raise ValueError('need at least three training draws')
    bm, rm = b.mean(axis=0), r.mean(axis=0)
    bs, rs = b.std(axis=0, ddof=1), r.std(axis=0, ddof=1)
    bs, rs = np.where(bs > 1e-12, bs, 1.), np.where(rs > 1e-12, rs, 1.)
    bz, rz = (b-bm)/bs, (r-rm)/rs
    coefficients = np.linalg.solve(bz.T@bz + len(b)*RIDGE*np.eye(b.shape[1]), bz.T@rz)
    joint = np.concatenate([bz, rz-bz@coefficients], axis=1)
    center = joint.mean(axis=0)
    cov_b, cov_j = np.cov(bz, rowvar=False), np.cov(joint, rowvar=False)
    cb = (1-SHRINKAGE)*cov_b + SHRINKAGE*np.eye(b.shape[1])
    cj = (1-SHRINKAGE)*cov_j + SHRINKAGE*np.eye(joint.shape[1])
    return {'baseline_mean':bm.tolist(), 'baseline_scale':bs.tolist(),
        'response_mean':rm.tolist(), 'response_scale':rs.tolist(),
        'coefficients':coefficients.tolist(), 'joint_center':center.tolist(),
        'baseline_precision':np.linalg.solve(cb, np.eye(len(cb))).tolist(),
        'joint_precision':np.linalg.solve(cj, np.eye(len(cj))).tolist(),
        'ridge':RIDGE, 'shrinkage':SHRINKAGE, 'training_draws':len(b),
        'condition_numbers':[float(np.linalg.cond(cb)), float(np.linalg.cond(cj))]}


def score_response_reference(baseline, response, model):
    b, r = checked_feature_pair(baseline, response)
    bz = (b-model['baseline_mean'])/model['baseline_scale']
    rz = (r-model['response_mean'])/model['response_scale']
    joint = np.concatenate([bz, rz-bz@model['coefficients']], axis=1)-model['joint_center']
    pb, pj = np.asarray(model['baseline_precision']), np.asarray(model['joint_precision'])
    return np.column_stack([np.einsum('bi,ij,bj->b', bz, pb, bz), np.einsum('bi,ij,bj->b', joint, pj, joint)])


def score_response_cuda(baseline, response, model, device='cuda:0'):
    import torch
    b, r = checked_feature_pair(baseline, response)
    tensor = lambda a:torch.as_tensor(a, dtype=torch.float64, device=device)
    bz = (tensor(b)-tensor(model['baseline_mean']))/tensor(model['baseline_scale'])
    rz = (tensor(r)-tensor(model['response_mean']))/tensor(model['response_scale'])
    joint = torch.cat([bz, rz-bz@tensor(model['coefficients'])], dim=1)-tensor(model['joint_center'])
    result = torch.stack([torch.einsum('bi,ij,bj->b', bz, tensor(model['baseline_precision']), bz),
        torch.einsum('bi,ij,bj->b', joint, tensor(model['joint_precision']), joint)], dim=1)
    return result.cpu().numpy()


def upper_tail_pvalues(scores, reference):
    s, ref = np.asarray(scores, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if s.ndim != 2 or ref.ndim != 2 or not len(ref) or s.shape[1] != ref.shape[1] or not np.all(np.isfinite(s)) or not np.all(np.isfinite(ref)):
        raise ValueError('need finite score matrices with matching columns')
    return np.stack([(1+np.sum(ref >= row, axis=0))/(len(ref)+1) for row in s])


def paired_win_pvalue(baseline_flags, combined_flags):
    a, b = np.asarray(baseline_flags, dtype=bool), np.asarray(combined_flags, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('need matching flag vectors')
    wins, losses = int(np.sum(b&~a)), int(np.sum(a&~b))
    n = wins+losses
    p = sum(math.comb(n, k) for k in range(wins, n+1))/2**n if n else 1.
    return wins, losses, float(p)
