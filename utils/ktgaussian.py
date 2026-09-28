# Copyright (c) 07/2024 Hao Li, Fudan University (h_li@fudan.edu.cn).
# All rights reserved.
# Translated to Python — source: https://github.com/CmrxRecon/CMRx4DFlow2026

import numpy as np


def create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y):
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
    y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
    X, Y = np.meshgrid(x, y)
    weight = np.exp(-(X ** 2 / (2 * sigma_x ** 2) + Y ** 2 / (2 * sigma_y ** 2)))
    return weight


def random_sampling_optimized(mask_size, total_points, weight, min_dist_lookup, existing_mask):
    width, height = mask_size
    sampled_points = []

    current_weight = weight * (1 - existing_mask)
    if np.sum(current_weight) <= 0:
        return np.array([])

    flat_weight = current_weight.ravel()
    prob = flat_weight / flat_weight.sum()

    forbidden_mask = np.zeros((height, width), dtype=bool)

    batch_size = max(total_points * 2, 1000)
    indices = np.random.choice(width * height, size=batch_size, p=prob)

    count = 0
    idx_ptr = 0
    Y_grid, X_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = indices[idx_ptr]
        idx_ptr += 1

        y, x = divmod(idx, width)

        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        sampled_points.append([x + 1, y + 1])
        count += 1

        d = float(min_dist_lookup[y, x])
        if d > 0:
            y_min, y_max = max(0, int(y - d)), min(height, int(y + d + 1))
            x_min, x_max = max(0, int(x - d)), min(width, int(x + d + 1))
            region_y = Y_grid[y_min:y_max, 0]
            region_x = X_grid[0, x_min:x_max]
            dist_sq = (region_y[:, np.newaxis] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= (dist_sq < d ** 2)

        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask) * (1 - forbidden_mask)
            sw = current_weight.sum()
            if sw <= 0:
                break
            prob = current_weight.ravel() / sw
            indices = np.random.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    return np.array(sampled_points)


def fun_mask_gen_2d(
    mask_size,
    total_points,
    pattern_num,
    sigma_x,
    sigma_y,
    min_dist_factor=3,
    rep_decay_factor=0.5,
    center_radius_x=0.5,
    center_radius_y=0.5,
):
    """
    Generate kt-Gaussian undersampling masks for the 2-D (ky, kz) phase-encoding plane.

    Parameters
    ----------
    mask_size     : (ky, kz)  — phase-encoding grid size
    total_points  : int       — sampled points per frame (= ky*kz // R)
    pattern_num   : int       — number of masks, one per cardiac frame (Nt)
    sigma_x/y     : float     — Gaussian std along ky/kz (use dim/5)
    min_dist_factor   : enforces Poisson-disc-like spacing (larger → more spread)
    rep_decay_factor  : < 1 discourages re-sampling same location across frames
                        (temporal incoherence — key for kt-MRI)
    center_radius_x/y : 0.5 → only the DC point is always sampled

    Returns
    -------
    masks : np.ndarray, shape (kz, ky, pattern_num), dtype float32
    """
    width, height = mask_size          # width=ky, height=kz
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)

    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()

    # outer k-space → large exclusion radius; center → small
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        center_ellipse = np.zeros((height, width), dtype=bool)
        center_ellipse[height // 2, width // 2] = True
    else:
        x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
        y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
        X, Y = np.meshgrid(x, y)
        center_ellipse = (X / center_radius_x) ** 2 + (Y / center_radius_y) ** 2 <= 1

    num_center_points = int(np.sum(center_ellipse))

    for p in range(pattern_num):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[center_ellipse] = 1

        needed = total_points - num_center_points
        if needed > 0:
            points = random_sampling_optimized(mask_size, needed, weight, min_dist_lookup, mask)
            if len(points) > 0:
                xs = points[:, 0].astype(int) - 1
                ys = points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        curr_total = int(np.sum(mask))
        if curr_total < total_points:
            extra = total_points - curr_total
            extra_points = random_sampling_optimized(mask_size, extra, weight, min_dist_lookup, mask)
            if len(extra_points) > 0:
                xs = extra_points[:, 0].astype(int) - 1
                ys = extra_points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        masks[:, :, p] = mask

    return masks


def ktgaussian_mask(ky, kz, T, usrate):
    """
    Wrapper that matches simulate_undersampling's interface but generates
    temporally-incoherent kt-Gaussian masks (different pattern per cardiac frame).

    Parameters
    ----------
    ky, kz  : phase-encoding grid dimensions
    T       : number of cardiac frames (pattern_num)
    usrate  : acceleration factor (e.g. 10, 20, 30, 40, 50)

    Returns
    -------
    mask : np.ndarray, shape (T, ky, kz), float32
           Different 2-D pattern per frame — broadcast over V and C yourself.
    """
    total_points = max(1, ky * kz // usrate)
    masks_kz_ky_t = fun_mask_gen_2d(
        mask_size=(ky, kz),
        total_points=total_points,
        pattern_num=T,
        sigma_x=ky / 5.0,
        sigma_y=kz / 5.0,
    )                                              # (kz, ky, T)
    return np.transpose(masks_kz_ky_t, (2, 1, 0)).astype(np.float32)  # (T, ky, kz)
