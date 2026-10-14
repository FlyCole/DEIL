import pdb
from typing import List
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution, lsq_linear, minimize
import open3d as o3d

def asymmetric_scale_icp(pcds: List[np.ndarray],
                         T_WCs: List[np.ndarray],
                         n_initialisations=10,
                         max_iterations=100,
                         tolerance=1e-6,
                         num_points=2048,
                         verbose: bool = False,
                         gt_first: bool = False):
    """
    Performs scale-only ICP to align two point clouds.

    Args:
        P1: NumPy array of shape (N, 3) representing the first point cloud.
        P2: NumPy array of shape (M, 3) representing the second point cloud.
        initial_scales: Tuple (s1, s2) representing the initial scaling factors.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        num_points: Number of points to downsample the point clouds to.

    Returns:
        Tuple (scaled_P1, scaled_P2, scales): Scaled point clouds and optimized scales.
    """

    def scale_pcd(pcd, T_WC, scale_x, scale_y, scale_z):
        #concatenate scales
        scale = np.array([scale_x, scale_y, scale_z])
        scaled_pcd = pcd * scale
        scaled_pcd -= T_WC[:3, 3] * (scale-1)
        return scaled_pcd
    
    pcds_og = [pcd.copy() for pcd in pcds]
    #downsample pcds to 2048 points
    for i, pcd in enumerate(pcds):
        if pcd.shape[0] > num_points:
            pcds[i] = pcd[np.random.choice(pcd.shape[0], num_points, replace=False)]

    for init_idx in range(n_initialisations):
        scales = np.random.uniform(0, 20, len(pcds)*3)
        prev_error = float('inf')
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            
            # Define the objective function for scale optimization
            def objective(scales):
                if gt_first:
                    scales[0:3] = 1
                # Scale each point cloud with its corresponding x, y, z scale factors
                scaled_pcds = [scale_pcd(pcd, T_WC, *scales[i*3 : i*3+3]) for i, (pcd, T_WC) in enumerate(zip(pcds, T_WCs))]
                error = None
                for idx in range(len(scaled_pcds)-1):
                    tree = KDTree(scaled_pcds[idx])
                    errors, _ = tree.query(scaled_pcds[idx+1])
                    if error is None:
                        error = errors
                    else:
                        error = np.concatenate((error, errors))

                error = np.sum(error) #TODO: maybe mean and then sum is faster
                if verbose:
                    print(f"Initialisation: {init_idx+1:02d}, Iteration: {iteration+1:03d}, Error: {error:.6f}")
                return error

            # Optimize the scales
            bounds = tuple([(0.01,None)] * len(pcds) * 3)
            result = minimize(objective, scales, method='L-BFGS-B', bounds=bounds)
            scales = result.x
            if gt_first:
                scales[0:3] = 1
            current_error = result.fun

            # Check for convergence
            if abs(prev_error - current_error) < tolerance:
                break

            prev_error = current_error

        if current_error < best_error:
            best_error = current_error
            best_scales = scales
    best_scaled_pcds = [scale_pcd(pcd, T_WC, *best_scales[i*3 : i*3+3]) for i, (pcd, T_WC) in enumerate(zip(pcds_og, T_WCs))]
    return best_scaled_pcds, best_scales



def scale_icp_sequential_gt(pcds: List[np.ndarray],
                            T_WCs: List[np.ndarray],
                            n_initialisations=10,
                            max_iterations=100,
                            tolerance=1e-6,
                            num_points=2048,
                            verbose: bool = False):
    """
    Performs scale-only ICP to align two point clouds.

    Args:
        P1: NumPy array of shape (N, 3) representing the first point cloud.
        P2: NumPy array of shape (M, 3) representing the second point cloud.
        initial_scales: Tuple (s1, s2) representing the initial scaling factors.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        num_points: Number of points to downsample the point clouds to.

    Returns:
        Tuple (scaled_P1, scaled_P2, scales): Scaled point clouds and optimized scales.
    """

    def scale_pcd(pcd, scale, T_WC):
        scaled_pcd = pcd * scale
        scaled_pcd -= T_WC[:3, 3] * (scale-1)
        return scaled_pcd
    
    pcds_og = [pcd.copy() for pcd in pcds]
    #downsample pcds to 2048 points
    for i, pcd in enumerate(pcds):
        if pcd.shape[0] > num_points:
            pcds[i] = pcd[np.random.choice(pcd.shape[0], num_points, replace=False)]

    best_scales = [1]

    for idx in range(len(pcds)-1):
        target_pcd = pcds[idx]
        source_pcd = pcds[idx+1]
        best_scales.append(1)
        for init_idx in range(n_initialisations):
            scale = np.random.uniform(0, 20, 1)
            prev_error = float('inf')
            best_error = float('inf')
            
            for iteration in range(max_iterations):
                
                # Define the objective function for scale optimization
                def objective(opt_scale):
                    scaled_pcd = scale_pcd(source_pcd, opt_scale[0], T_WCs[idx+1])
                    
                    tree = KDTree(target_pcd)
                    errors, _ = tree.query(scaled_pcd)
                    error = np.sum(errors)
                    if verbose:
                        print(f"Initialisation: {init_idx+1:02d}, Iteration: {iteration+1:03d}, Error: {error:.6f}")
                    return error

                # Optimize the scales
                bounds = ((0.0000001,None), )
                result = minimize(objective, scale, method='L-BFGS-B', bounds=bounds)
                scale = result.x
                current_error = result.fun

                # Check for convergence
                if abs(prev_error - current_error) < tolerance:
                    break

                prev_error = current_error

            if current_error < best_error:
                best_error = current_error
                best_scales[-1] = scale[0]
    best_scaled_pcds = [scale_pcd(pcd, scale, T_WC) for pcd, scale, T_WC in zip(pcds_og, best_scales, T_WCs)]
    return best_scaled_pcds, best_scales


def scale_icp_old(pcds: List[np.ndarray],
              T_WCs: List[np.ndarray],
              n_initialisations=10,
              max_iterations=100,
              tolerance=1e-6,
              num_points=2048,
              previous_scale=None,
              gt_first: bool = False,
              intrinsics=None,
              visualise=False,
              verbose: bool = False):
    """
    Performs scale-only ICP to align two point clouds.

    Args:
        P1: NumPy array of shape (N, 3) / (N, 6) representing the first point cloud.
        P2: NumPy array of shape (M, 3) / (M, 6) representing the second point cloud.
        initial_scales: Tuple (s1, s2) representing the initial scaling factors.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        num_points: Number of points to downsample the point clouds to.

    Returns:
        Tuple (scaled_P1, scaled_P2, scales): Scaled point clouds and optimized scales.
    """

    def scale_pcd(pcd, scale, T_WC):
        scaled_pcd = pcd * scale
        scaled_pcd -= T_WC[:3, 3] * (scale-1)
        return scaled_pcd

    color_mode = False
    if pcds[0].shape[1] == 6:
        # Separate colors if present
        pcds_colors = [pcd[:, 3:6] for pcd in pcds]
        pcds = [pcd[:, :3] for pcd in pcds]
        color_mode = True

    pcds_og = [np.copy(pcd) for pcd in pcds]

    # Downsample pcds to 2048 points
    for i, pcd in enumerate(pcds):
        if pcd.shape[0] > num_points:
            pcds[i] = pcd[np.random.choice(pcd.shape[0], num_points, replace=False)]
        elif pcd.shape[0] == 0:
            # TODO: handle empty point clouds better
            pcds[i] = np.zeros((num_points, 3))

    best_error = float('inf')
    best_scales = None

    for init_idx in range(n_initialisations):
        if previous_scale is not None:
            scales = previous_scale
        else:
            scales = np.random.uniform(0, 10, len(pcds))
        prev_error = float('inf')
        
        # TODO: change
        # if previous_scale is not None:
        #     fix_scale = scales[0]
        
        for iteration in range(max_iterations):
            
            # Define the objective function for scale optimization
            def objective(scales):
                if gt_first:
                    scales[1] = 1
                # TODO: change
                # if previous_scale is not None:
                #     scales[0] = fix_scale

                scaled_pcds = [scale_pcd(pcd, scale, T_WC) for pcd, scale, T_WC in zip(pcds, scales, T_WCs)]
                error = None
                for idx in range(len(scaled_pcds)-1):
                    tree = KDTree(scaled_pcds[idx])
                    errors, _ = tree.query(scaled_pcds[idx+1])
                    if error is None:
                        error = errors
                    else:
                        error = np.concatenate((error, errors))

                error = np.sum(error) #TODO: maybe mean and then sum is faster
                if verbose:
                    print(f"Initialisation: {init_idx+1:02d}, Iteration: {iteration+1:03d}, Error: {error:.6f}")
                return error

            # Optimize the scales
            bounds = tuple([(0.0000001,None)] * len(pcds))
            result = minimize(objective, scales, method='L-BFGS-B', bounds=bounds)
            scales = result.x
            if gt_first:
                scales[1] = 1
            current_error = result.fun

            # Check for convergence
            if abs(prev_error - current_error) < tolerance:
                break

            prev_error = current_error

        if current_error < best_error:
            best_error = current_error
            best_scales = scales

    if best_scales is None:
        raise RuntimeError("scale_icp failed to find scales. Check n_initialisations and input point clouds.")
    best_scaled_pcds = [scale_pcd(pcd, scale, T_WC) for pcd, scale, T_WC in zip(pcds_og, best_scales, T_WCs)]
    if color_mode:
        # Reattach colors
        for i in range(len(best_scaled_pcds)):
            best_scaled_pcds[i] = np.hstack((best_scaled_pcds[i], pcds_colors[i]))

    return best_scaled_pcds, best_scales


def scale_icp(pcds: List[np.ndarray],
                  T_WCs: List[np.ndarray],
                  n_initialisations=10,
                  max_iterations=100,
                  tolerance=1e-5,
                  num_points=512,
                  previous_scale=None,
                  gt_first: bool = False,
                  intrinsics=None,
                  visualise=False,
                  verbose: bool = False):
    """Align point clouds by estimating one positive scale per cloud.

    Points are scaled about their camera centers in world coordinates. It uses a
    symmetric, 80%-trimmed Chamfer loss, searches in log-scale with
    differential evolution, alternates fixed-correspondence least-squares
    scale updates, and finishes with derivative-free Powell refinement.

    ``gt_first=True`` fixes the *first* point cloud's scale to 1 without
    exposing that fixed value to an optimizer.  ``previous_scale``, when
    supplied, seeds the global search and expands the default scale bounds if
    necessary.  ``intrinsics`` and ``visualise`` are accepted for API
    compatibility with older callers and are not used by this implementation.

    Args:
        pcds: Point clouds with shape ``(N, 3)`` or ``(N, 6)``.  For six-column
            inputs, the last three color columns are preserved in the output.
        T_WCs: Camera-to-world transforms corresponding to ``pcds``.
        n_initialisations: Minimum global-search population target.  Larger
            values improve coverage but increase runtime.
        max_iterations: Iteration limit for each optimization stage.
        tolerance: Absolute/relative convergence tolerance.
        num_points: Maximum points per cloud used during optimization.
        previous_scale: Optional scalar or one-dimensional scale estimate.
        gt_first: Fix the first scale to 1 when true.
        intrinsics: Accepted for compatibility; unused.
        visualise: Accepted for compatibility; unused.
        verbose: Print optimizer progress and the final loss when true.

    Returns:
        ``(scaled_pcds, scales)``: aligned world-frame clouds and per-camera scales.
    """

    # These finite bounds prevent collapse and unbounded global exploration.
    # If a previous estimate falls outside them, the bounds are expanded below.
    default_min_scale = 0.01
    default_max_scale = 20.0
    trim_fraction = 0.80

    if len(pcds) < 2:
        raise ValueError("scale_icp_new requires at least two point clouds.")
    if len(pcds) != len(T_WCs):
        raise ValueError("pcds and T_WCs must have the same length.")
    if n_initialisations < 1:
        raise ValueError("n_initialisations must be at least 1.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if num_points < 1:
        raise ValueError("num_points must be at least 1.")

    # Keep these arguments in the signature so callers can switch between the
    # old and new implementations without changing call sites.
    _ = intrinsics, visualise

    xyz_pcds = []
    pcd_colors = []
    color_mode = None
    for idx, pcd in enumerate(pcds):
        pcd = np.asarray(pcd)
        if pcd.ndim != 2 or pcd.shape[1] not in (3, 6):
            raise ValueError(
                f"pcds[{idx}] must have shape (N, 3) or (N, 6); got {pcd.shape}."
            )
        has_color = pcd.shape[1] == 6
        if color_mode is None:
            color_mode = has_color
        elif has_color != color_mode:
            raise ValueError("All point clouds must use the same number of columns.")
        xyz_pcds.append(np.asarray(pcd[:, :3], dtype=np.float64).copy())
        if has_color:
            pcd_colors.append(pcd[:, 3:6].copy())

    translations = []
    for idx, T_WC in enumerate(T_WCs):
        T_WC = np.asarray(T_WC, dtype=np.float64)
        if T_WC.ndim != 2 or T_WC.shape[0] < 3 or T_WC.shape[1] < 4:
            raise ValueError(
                f"T_WCs[{idx}] must contain at least a 3x4 transform; got {T_WC.shape}."
            )
        translation = T_WC[:3, 3]
        if not np.all(np.isfinite(translation)):
            raise ValueError(f"T_WCs[{idx}] contains a non-finite translation.")
        translations.append(translation)

    # Optimize only with finite points and do not mutate the caller's arrays.
    sampled_pcds = []
    for idx, pcd in enumerate(xyz_pcds):
        finite_pcd = pcd[np.all(np.isfinite(pcd), axis=1)]
        if finite_pcd.shape[0] == 0:
            raise ValueError(f"pcds[{idx}] has no finite XYZ points.")
        if finite_pcd.shape[0] > num_points:
            selected = np.random.choice(
                finite_pcd.shape[0], num_points, replace=False
            )
            finite_pcd = finite_pcd[selected]
        sampled_pcds.append(finite_pcd)

    num_clouds = len(sampled_pcds)
    lower_scales = np.full(num_clouds, default_min_scale, dtype=np.float64)
    upper_scales = np.full(num_clouds, default_max_scale, dtype=np.float64)

    initial_scales = None
    if previous_scale is not None:
        initial_scales = np.asarray(previous_scale, dtype=np.float64)
        if initial_scales.ndim == 0:
            initial_scales = np.full(num_clouds, float(initial_scales))
        else:
            initial_scales = initial_scales.reshape(-1).copy()
        if initial_scales.size != num_clouds:
            raise ValueError(
                "previous_scale must be a scalar or contain one value per point cloud."
            )
        if not np.all(np.isfinite(initial_scales)) or np.any(initial_scales <= 0):
            raise ValueError("previous_scale values must be finite and positive.")
        lower_scales = np.minimum(lower_scales, initial_scales / 10.0)
        upper_scales = np.maximum(upper_scales, initial_scales * 10.0)

    fixed_indices = np.array([0], dtype=int) if gt_first else np.empty(0, dtype=int)
    free_indices = np.arange(1, num_clouds) if gt_first else np.arange(num_clouds)
    if gt_first and initial_scales is not None:
        initial_scales[0] = 1.0

    def scale_pcd(pcd, scale, translation):
        # Equivalent to translation + scale * (pcd - translation): scaling is
        # performed about the camera center rather than the world origin.
        return translation + scale * (pcd - translation)

    def decode_log_scales(log_free_scales):
        scales = np.ones(num_clouds, dtype=np.float64)
        scales[free_indices] = np.exp(log_free_scales)
        return scales

    def trimmed_mean_squared(distances):
        keep_count = max(1, int(np.ceil(trim_fraction * distances.size)))
        kept = np.partition(distances, keep_count - 1)[:keep_count]
        return float(np.mean(kept * kept))

    def objective(log_free_scales):
        scales = decode_log_scales(log_free_scales)
        scaled = [
            scale_pcd(pcd, scale, translation)
            for pcd, scale, translation in zip(sampled_pcds, scales, translations)
        ]

        pair_losses = []
        for left, right in zip(scaled[:-1], scaled[1:]):
            left_tree = KDTree(left)
            right_tree = KDTree(right)
            right_to_left, _ = left_tree.query(right)
            left_to_right, _ = right_tree.query(left)
            pair_losses.append(
                0.5
                * (
                    trimmed_mean_squared(right_to_left)
                    + trimmed_mean_squared(left_to_right)
                )
            )
        return float(np.mean(pair_losses))

    log_bounds = [
        (np.log(lower_scales[idx]), np.log(upper_scales[idx]))
        for idx in free_indices
    ]

    # Differential evolution supplies global exploration.  scipy's popsize is
    # a multiplier, so choose one that gives at least n_initialisations members
    # while maintaining a useful population in higher dimensions.
    population_multiplier = max(
        5, int(np.ceil(n_initialisations / max(1, free_indices.size)))
    )
    global_kwargs = {}
    if initial_scales is not None:
        clipped_initial = np.clip(
            initial_scales[free_indices],
            lower_scales[free_indices],
            upper_scales[free_indices],
        )
        global_kwargs["x0"] = np.log(clipped_initial)

    global_result = differential_evolution(
        objective,
        bounds=log_bounds,
        maxiter=max_iterations,
        popsize=population_multiplier,
        tol=tolerance,
        atol=tolerance,
        polish=False,
        disp=verbose,
        **global_kwargs,
    )

    current_scales = decode_log_scales(global_result.x)
    current_error = objective(np.log(current_scales[free_indices]))

    def trimmed_correspondence_indices(distances):
        keep_count = max(1, int(np.ceil(trim_fraction * distances.size)))
        return np.argpartition(distances, keep_count - 1)[:keep_count]

    def fixed_correspondence_system(scales):
        """Build A @ scales ~= b for current symmetric correspondences."""
        scaled = [
            scale_pcd(pcd, scale, translation)
            for pcd, scale, translation in zip(sampled_pcds, scales, translations)
        ]
        matrix_blocks = []
        rhs_blocks = []
        num_pairs = num_clouds - 1

        def append_matches(left_idx, left_points, right_idx, right_points):
            count = left_points.shape[0]
            block = np.zeros((count * 3, num_clouds), dtype=np.float64)
            block[:, left_idx] = (
                left_points - translations[left_idx]
            ).reshape(-1)
            block[:, right_idx] = -(
                right_points - translations[right_idx]
            ).reshape(-1)
            rhs = np.tile(
                translations[right_idx] - translations[left_idx], (count, 1)
            ).reshape(-1)

            # Match the pair/direction averaging used by objective().
            weight = np.sqrt(1.0 / (2.0 * num_pairs * count))
            matrix_blocks.append(block * weight)
            rhs_blocks.append(rhs * weight)

        for left_idx in range(num_pairs):
            right_idx = left_idx + 1
            left_tree = KDTree(scaled[left_idx])
            right_tree = KDTree(scaled[right_idx])

            right_distances, matched_left = left_tree.query(scaled[right_idx])
            kept_right = trimmed_correspondence_indices(right_distances)
            append_matches(
                left_idx,
                sampled_pcds[left_idx][matched_left[kept_right]],
                right_idx,
                sampled_pcds[right_idx][kept_right],
            )

            left_distances, matched_right = right_tree.query(scaled[left_idx])
            kept_left = trimmed_correspondence_indices(left_distances)
            append_matches(
                left_idx,
                sampled_pcds[left_idx][kept_left],
                right_idx,
                sampled_pcds[right_idx][matched_right[kept_left]],
            )

        return np.vstack(matrix_blocks), np.concatenate(rhs_blocks)

    # ICP-style alternating updates: correspondences are held fixed while the
    # bounded scale subproblem is solved exactly as a linear least-squares task.
    for iteration in range(max_iterations):
        full_matrix, rhs = fixed_correspondence_system(current_scales)
        if fixed_indices.size:
            rhs = rhs - full_matrix[:, fixed_indices] @ current_scales[fixed_indices]
        free_matrix = full_matrix[:, free_indices]
        least_squares_result = lsq_linear(
            free_matrix,
            rhs,
            bounds=(lower_scales[free_indices], upper_scales[free_indices]),
            tol=tolerance,
            max_iter=max_iterations,
        )

        candidate_scales = current_scales.copy()
        candidate_scales[free_indices] = least_squares_result.x
        candidate_error = objective(np.log(candidate_scales[free_indices]))

        # Changing correspondences can make the full Chamfer loss increase even
        # though the fixed-correspondence subproblem improved.  Backtrack in
        # log-space so this stage never accepts a worse solution.
        if candidate_error > current_error:
            current_log = np.log(current_scales[free_indices])
            candidate_log = np.log(candidate_scales[free_indices])
            accepted = False
            for step_size in (0.5, 0.25, 0.125, 0.0625):
                trial_log = current_log + step_size * (candidate_log - current_log)
                trial_error = objective(trial_log)
                if trial_error < current_error:
                    candidate_scales = decode_log_scales(trial_log)
                    candidate_error = trial_error
                    accepted = True
                    break
            if not accepted:
                break

        improvement = current_error - candidate_error
        scale_change = np.max(
            np.abs(
                np.log(candidate_scales[free_indices])
                - np.log(current_scales[free_indices])
            )
        )
        current_scales = candidate_scales
        current_error = candidate_error
        if verbose:
            print(
                f"Fixed-correspondence iteration {iteration + 1:03d}: "
                f"error={current_error:.6g}"
            )
        if improvement <= tolerance * max(1.0, current_error) or scale_change <= tolerance:
            break

    # Powell is derivative-free and therefore handles nearest-neighbor switches
    # better than a finite-difference quasi-Newton method such as L-BFGS-B.
    local_result = minimize(
        objective,
        np.log(current_scales[free_indices]),
        method="Powell",
        bounds=log_bounds,
        options={
            "maxiter": max_iterations,
            "xtol": tolerance,
            "ftol": tolerance,
            "disp": verbose,
        },
    )
    if local_result.fun < current_error:
        current_scales = decode_log_scales(local_result.x)
        current_error = float(local_result.fun)

    best_scaled_pcds = [
        scale_pcd(pcd, scale, translation)
        for pcd, scale, translation in zip(xyz_pcds, current_scales, translations)
    ]
    if color_mode:
        best_scaled_pcds = [
            np.hstack((scaled_pcd, colors))
            for scaled_pcd, colors in zip(best_scaled_pcds, pcd_colors)
        ]

    if verbose:
        print(f"scale_icp_new final error={current_error:.6g}, scales={current_scales}")
    return best_scaled_pcds, current_scales


def scale_icp_full(pcds: List[np.ndarray],
                   T_WCs: List[np.ndarray],
                   max_iterations=100,
                   tolerance=1e-6,
                   num_points=2048):
    """
    Performs scale-only ICP to align two point clouds.

    Args:
        P1: NumPy array of shape (N, 3) representing the first point cloud.
        P2: NumPy array of shape (M, 3) representing the second point cloud.
        initial_scales: Tuple (s1, s2) representing the initial scaling factors.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        num_points: Number of points to downsample the point clouds to.

    Returns:
        Tuple (scaled_P1, scaled_P2, scales): Scaled point clouds and optimized scales.
    """

    def scale_pcd(pcd, scale, T_WC):
        scaled_pcd = pcd * scale
        scaled_pcd -= T_WC[:3, 3] * (scale-1)
        return scaled_pcd
    
    pcds_og = [pcd.copy() for pcd in pcds]
    #downsample pcds to 2048 points
    for i, pcd in enumerate(pcds):
        if pcd.shape[0] > num_points:
            pcds[i] = pcd[np.random.choice(pcd.shape[0], num_points, replace=False)]

    for _ in range(10):
        scales = np.random.uniform(0, 20, len(pcds))
        prev_error = float('inf')
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            
            # Define the objective function for scale optimization
            def objective(scales):
                scaled_pcds = [scale_pcd(pcd, scale, T_WC) for pcd, scale, T_WC in zip(pcds, scales, T_WCs)]
                error = None
                for ref_pcd in scaled_pcds:
                    tree = KDTree(ref_pcd)
                    for pcd in scaled_pcds:
                        if pcd is ref_pcd:
                            continue
                        errors, _ = tree.query(pcd)
                        if error is None:
                            error = errors
                        else:
                            error = np.concatenate((error, errors))

                error = np.mean(error) #TODO: maybe mean and then sum is faster
                print(f"Error: {error}")
                return error

            # Optimize the scales
            bounds = tuple([(0.0000001,None)] * len(pcds))
            result = minimize(objective, scales, method='L-BFGS-B', bounds=bounds)
            scales = result.x
            current_error = result.fun

            # Check for convergence
            if abs(prev_error - current_error) < tolerance:
                break

            prev_error = current_error

        if current_error < best_error:
            best_error = current_error
            best_scales = scales
    best_scaled_pcds = [scale_pcd(pcd, scale, T_WC) for pcd, scale, T_WC in zip(pcds_og, best_scales, T_WCs)]
    return best_scaled_pcds, best_scales

def scale_2pcd_icp(pcd1, pcd2, T_WC1, T_WC2, max_iterations=100, tolerance=1e-6, num_points=2048):
    """
    Performs scale-only ICP to align two point clouds.

    Args:
        P1: NumPy array of shape (N, 3) representing the first point cloud.
        P2: NumPy array of shape (M, 3) representing the second point cloud.
        initial_scales: Tuple (s1, s2) representing the initial scaling factors.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
        num_points: Number of points to downsample the point clouds to.

    Returns:
        Tuple (scaled_P1, scaled_P2, scales): Scaled point clouds and optimized scales.
    """

    def scale_pcd(pcd, scale, T_WC):
        scaled_pcd = pcd * scale
        scaled_pcd -= T_WC[:3, 3] * (scale-1)
        return scaled_pcd
    
    pcd1_og = pcd1.copy()
    pcd2_og = pcd2.copy()
    # Downsample pcd1 to 2048 points
    if pcd1.shape[0] > num_points:
        pcd1 = pcd1[np.random.choice(pcd1.shape[0], num_points, replace=False)]
    # Downsample pcd2 to 2048 points
    if pcd2.shape[0] > num_points:
        pcd2 = pcd2[np.random.choice(pcd2.shape[0], num_points, replace=False)]

    for _ in range(10):
        s1, s2 = np.random.uniform(0, 20, 2)
        prev_error = float('inf')
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            
            # Define the objective function for scale optimization
            def objective(scales):
                s1_opt, s2_opt = scales
                scaled_P1_opt = scale_pcd(pcd1, s1_opt, T_WC1)
                scaled_P2_opt = scale_pcd(pcd2, s2_opt, T_WC2)

                tree1 = KDTree(scaled_P2_opt)
                error1, _ = tree1.query(scaled_P1_opt)
                tree2 = KDTree(scaled_P1_opt)
                error2, _ = tree2.query(scaled_P2_opt)

                error = np.mean(np.concatenate((error1, error2))) #TODO: maybe mean and then sum is faster
                print(f"Error: {error}")
                return error

            # Optimize the scales
            result = minimize(objective, (s1, s2), method='L-BFGS-B', bounds=((0.0000001,None),(0.0000001,None)))
            s1, s2 = result.x
            current_error = result.fun

            # Check for convergence
            if abs(prev_error - current_error) < tolerance:
                break

            prev_error = current_error

        if current_error < best_error:
            best_error = current_error
            best_s1 = s1
            best_s2 = s2
            
    return scale_pcd(pcd1_og, best_s1, T_WC1), scale_pcd(pcd2_og, best_s2, T_WC2), (best_s1, best_s2)
