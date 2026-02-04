import numpy as np
import torch
import matplotlib.pyplot as plt


SKELETON_EDGES = [
    (0, 1), (1, 2),  # 頭部
    (2, 3), (3, 4), (4, 5), (5, 6),  # 右腕
    (2, 7), (7, 8), (8, 9), (9, 10),  # 左腕
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # 脊椎
    (15, 16), (16, 17), (17, 18),  # 右足
    (15, 19), (19, 20), (20, 21),  # 左足
]

PLOT_COLORS = {
    "original_cloud": "#00008B",
    "original_edge": "#00008B",
    "visible_joint": "#1f77b4",
    "visible_edge": "#1f77b4",
    "masked_joint": "#c7c7c7",
    "masked_placeholder": "#ececec",
    "masked_edge": "#d2d2d2",
    "reconstructed_joint": "#2ca02c",
    "reconstructed_edge": "#41ab5d",
}

DEFAULT_VIEW = {
    "elev": 180.0,
    "azim": 180.0,
}


def to_numpy_indices(indices):
    """Convert masked joint indices to a numpy array."""
    if indices is None:
        return np.array([], dtype=int)

    if isinstance(indices, (list, tuple)):
        if len(indices) == 0:
            return np.array([], dtype=int)
        # 可視化用途なので、先頭フレームのマスクを代表として使用
        return to_numpy_indices(indices[0])

    tensor = collapse_mask_indices(indices)
    if tensor.numel() == 0:
        return np.array([], dtype=int)
    return tensor.numpy().astype(int)


def to_numpy_array(array_like):
    """Safely convert tensors or other iterables to a numpy array."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _add_legend_entry(legend_entries, handle, label):
    if handle is None or label is None:
        return
    legend_entries.append((handle, label))


def _unique_legend_entries(legend_entries):
    seen = set()
    handles = []
    labels = []
    for handle, label in legend_entries:
        if label in seen:
            continue
        handles.append(handle)
        labels.append(label)
        seen.add(label)
    return handles, labels


def collapse_mask_indices(mask_indices_sample):
    """
    Convert mask indices for a single sample into a unique 1-D CPU tensor.
    Supports per-frame lists as well as flat tensors/lists.
    """
    if mask_indices_sample is None:
        return torch.empty(0, dtype=torch.long)

    if torch.is_tensor(mask_indices_sample):
        return mask_indices_sample.detach().cpu().view(-1)

    if isinstance(mask_indices_sample, np.ndarray):
        return torch.from_numpy(mask_indices_sample.astype(np.int64)).view(-1)

    if isinstance(mask_indices_sample, (list, tuple)):
        tensors = []
        for item in mask_indices_sample:
            t = collapse_mask_indices(item)
            if t.numel() > 0:
                tensors.append(t.view(-1))
        if not tensors:
            return torch.empty(0, dtype=torch.long)
        concatenated = torch.cat(tensors)
        return torch.unique(concatenated)

    return torch.tensor([int(mask_indices_sample)], dtype=torch.long)


def get_frame_mask_indices(mask_indices, sample_idx=0, frame_idx=0):
    """
    Return a CPU long tensor of masked joints for the specified sample/frame.
    Falls back gracefully if indices are stored in legacy formats.
    """
    if mask_indices is None:
        return torch.empty(0, dtype=torch.long)

    target = mask_indices
    if isinstance(target, (list, tuple)):
        if len(target) == 0:
            return torch.empty(0, dtype=torch.long)
        sample_idx = max(min(sample_idx, len(target) - 1), 0)
        target = target[sample_idx]
        if isinstance(target, (list, tuple)):
            if len(target) == 0:
                return torch.empty(0, dtype=torch.long)
            frame_idx = max(min(frame_idx, len(target) - 1), 0)
            target = target[frame_idx]

    return collapse_mask_indices(target)


def get_frame_mask_numpy(mask_indices, sample_idx=0, frame_idx=0):
    """Helper that returns masked joint indices as a numpy array for a frame."""
    tensor = get_frame_mask_indices(mask_indices, sample_idx=sample_idx, frame_idx=frame_idx)
    if tensor.numel() == 0:
        return np.array([], dtype=int)
    return tensor.numpy().astype(int)


def _ensure_long_tensor(indices, device):
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long).view(-1)
    if isinstance(indices, (list, tuple, np.ndarray)):
        if len(indices) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(indices, device=device, dtype=torch.long).view(-1)
    return torch.tensor([int(indices)], device=device, dtype=torch.long)


def to_camera_coords(points):
    """
    JTA座標 (x=右+, y=上+, z=奥+) を matplotlib でのカメラ視点可視化用に変換する。
    """
    points = np.asarray(points)
    x_down = -points[..., 0]
    y_left = -points[..., 1]
    z_forward = points[..., 2]
    return x_down, y_left, z_forward


def _collect_axis_values(arrays):
    flattened = []
    for arr in arrays:
        if arr is None:
            continue
        np_arr = np.asarray(arr)
        if np_arr.size == 0:
            continue
        flattened.append(np_arr.reshape(-1))
    if not flattened:
        return np.array([0.0])
    return np.concatenate(flattened)


def _calculate_axis_bounds(x_arrays, y_arrays, z_arrays, margin):
    """Calculate axis limits and aspect ratios for a collection of coordinates."""
    x_vals = _collect_axis_values(x_arrays)
    y_vals = _collect_axis_values(y_arrays)
    z_vals = _collect_axis_values(z_arrays)

    def _limits(values):
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        span = max(max_val - min_val, 1e-6)
        pad = span * margin
        return (min_val - pad, max_val + pad), max(span, 1e-6)

    x_lim, x_span = _limits(x_vals)
    y_lim, y_span = _limits(y_vals)
    z_lim, z_span = _limits(z_vals)

    return {
        "x_lim": x_lim,
        "y_lim": y_lim,
        "z_lim": z_lim,
        "box_aspect": (x_span, y_span, z_span),
    }


def set_axes_equal(ax, x_arrays, y_arrays, z_arrays, margin=0.05, bounds=None):
    """Set equal scale on all axes while keeping a configurable margin."""
    if bounds is None:
        bounds = _calculate_axis_bounds(x_arrays, y_arrays, z_arrays, margin)

    ax.set_xlim(*bounds["x_lim"])
    ax.set_ylim(*bounds["y_lim"])
    ax.set_zlim(*bounds["z_lim"])
    ax.set_box_aspect(bounds["box_aspect"])
    return bounds


def compute_coord_axis_bounds(coord_sets, margin=0.08):
    """
    Compute axis bounds across multiple coordinate tuples (x, y, z).
    """
    x_arrays, y_arrays, z_arrays = [], [], []

    def _accumulate(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 3 and all(hasattr(arr, "__array__") or isinstance(arr, np.ndarray) for arr in item):
                x_arrays.append(np.asarray(item[0]))
                y_arrays.append(np.asarray(item[1]))
                z_arrays.append(np.asarray(item[2]))
            else:
                for sub_item in item:
                    _accumulate(sub_item)

    _accumulate(coord_sets)

    if not x_arrays:
        return None

    return _calculate_axis_bounds(x_arrays, y_arrays, z_arrays, margin)


def set_camera_view(ax, elev=None, azim=None):
    """Apply a consistent front-facing camera view without perspective distortion."""
    elev = DEFAULT_VIEW["elev"] if elev is None else elev
    azim = DEFAULT_VIEW["azim"] if azim is None else azim
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        # Older matplotlib versions may not support set_proj_type
        pass


def _setup_axis(ax, title=None):
    ax.set_facecolor("#fbfbfb")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title is not None:
        ax.set_title(title)


def _scatter_points(
    ax,
    coords,
    indices=None,
    *,
    color,
    size,
    marker="o",
    alpha=1.0,
    edgecolor=None,
    linewidth=0.0,
):
    x, y, z = coords
    if indices is None:
        xs, ys, zs = x, y, z
    else:
        xs, ys, zs = x[indices], y[indices], z[indices]
    ax.scatter(
        xs,
        ys,
        zs,
        color=color,
        s=size,
        marker=marker,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidths=linewidth,
        depthshade=False,
    )


def _draw_edges(ax, coords, *, predicate=None, color, linewidth, alpha):
    x, y, z = coords
    joint_count = len(x)
    for i, j in SKELETON_EDGES:
        if i >= joint_count or j >= joint_count:
            continue
        if predicate is not None and not predicate(i, j):
            continue
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [z[i], z[j]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _mask_observed_coords(mask_coords_raw, orig_coords, masked_ids):
    if masked_ids.size == 0:
        return mask_coords_raw
    masked = [arr.copy() for arr in mask_coords_raw]
    for dst, src in zip(masked, orig_coords):
        dst[masked_ids] = src[masked_ids]
    return tuple(masked)


def _subset_coords(coords, indices):
    if indices.size == 0:
        return None
    return tuple(arr[indices] for arr in coords)


def _finalize_axis(ax, coord_sets, margin=0.08, bounds=None):
    x_arrays, y_arrays, z_arrays = [], [], []
    for coords in coord_sets:
        if coords is None:
            continue
        x_arrays.append(coords[0])
        y_arrays.append(coords[1])
        z_arrays.append(coords[2])
    if not x_arrays:
        return
    set_axes_equal(ax, x_arrays, y_arrays, z_arrays, margin=margin, bounds=bounds)
    set_camera_view(ax)


def _draw_original_view(ax, orig_coords, *, title, point_size=40, edge_width=1.2, alpha=0.85, axis_bounds=None):
    _setup_axis(ax, title)
    _scatter_points(ax, orig_coords, color=PLOT_COLORS["original_cloud"], size=point_size, alpha=alpha)
    _draw_edges(
        ax,
        orig_coords,
        color=PLOT_COLORS["original_edge"],
        linewidth=edge_width,
        alpha=0.6,
    )
    _finalize_axis(ax, [orig_coords], bounds=axis_bounds)


def _draw_mask_view(
    ax,
    orig_coords,
    mask_coords,
    mask_coords_raw,
    masked_ids,
    unmasked_ids,
    *,
    title,
    visible_size=55,
    masked_size=60,
    token_size=34,
    visible_edge_width=1.8,
    masked_edge_width=0.9,
    axis_bounds=None,
):
    _setup_axis(ax, title)
    if unmasked_ids.size > 0:
        _scatter_points(
            ax,
            mask_coords,
            unmasked_ids,
            color=PLOT_COLORS["visible_joint"],
            size=visible_size,
            alpha=0.95,
        )
    if masked_ids.size > 0:
        _scatter_points(
            ax,
            mask_coords_raw,
            masked_ids,
            color=PLOT_COLORS["masked_placeholder"],
            size=token_size,
            marker="x",
            alpha=0.28,
            linewidth=0.6,
        )
        _scatter_points(
            ax,
            orig_coords,
            masked_ids,
            color=PLOT_COLORS["masked_joint"],
            size=masked_size,
            marker="o",
            alpha=0.38,
            edgecolor=None,
            linewidth=0.0,
        )
    masked_set = set(masked_ids.tolist())
    unmasked_set = set(unmasked_ids.tolist())
    _draw_edges(
        ax,
        orig_coords,
        predicate=lambda i, j: i in masked_set and j in masked_set,
        color=PLOT_COLORS["masked_edge"],
        linewidth=masked_edge_width,
        alpha=0.28,
    )
    _draw_edges(
        ax,
        mask_coords,
        predicate=lambda i, j: i in unmasked_set and j in unmasked_set,
        color=PLOT_COLORS["visible_edge"],
        linewidth=visible_edge_width,
        alpha=0.85,
    )
    _finalize_axis(
        ax,
        [
            orig_coords,
            mask_coords,
            _subset_coords(mask_coords_raw, masked_ids),
        ],
        bounds=axis_bounds,
    )


def _draw_reconstructed_plain_view(
    ax,
    recon_coords,
    *,
    title,
    point_size=55,
    edge_width=1.6,
    alpha=0.9,
    axis_bounds=None,
):
    _setup_axis(ax, title)
    _scatter_points(
        ax,
        recon_coords,
        color=PLOT_COLORS["reconstructed_joint"],
        size=point_size,
        alpha=alpha,
    )
    _draw_edges(
        ax,
        recon_coords,
        color=PLOT_COLORS["reconstructed_edge"],
        linewidth=edge_width,
        alpha=0.8,
    )
    _finalize_axis(ax, [recon_coords], bounds=axis_bounds)


def _draw_reconstructed_error_view(
    ax,
    orig_coords,
    recon_coords,
    masked_ids,
    unmasked_ids,
    *,
    title,
    visible_size=60,
    masked_size=80,
    dashed_alpha=0.6,
    dashed_width=0.9,
    orig_alpha=0.25,
    axis_bounds=None,
):
    _setup_axis(ax, title)
    _scatter_points(
        ax,
        orig_coords,
        color=PLOT_COLORS["original_cloud"],
        size=28,
        alpha=orig_alpha,
    )
    _draw_edges(
        ax,
        orig_coords,
        color=PLOT_COLORS["original_edge"],
        linewidth=1.0,
        alpha=0.35,
    )
    if unmasked_ids.size > 0:
        _scatter_points(
            ax,
            recon_coords,
            unmasked_ids,
            color=PLOT_COLORS["reconstructed_joint"],
            size=visible_size,
            marker="o",
            alpha=0.9,
        )
    if masked_ids.size > 0:
        _scatter_points(
            ax,
            recon_coords,
            masked_ids,
            color=PLOT_COLORS["reconstructed_joint"],
            size=masked_size,
            marker="^",
            alpha=0.95,
        )
    x, y, z = orig_coords
    xr, yr, zr = recon_coords
    for idx in range(len(x)):
        ax.plot(
            [x[idx], xr[idx]],
            [y[idx], yr[idx]],
            [z[idx], zr[idx]],
            color="#7f7f7f",
            linestyle="--",
            linewidth=dashed_width,
            alpha=dashed_alpha,
        )
    _draw_edges(
        ax,
        recon_coords,
        color=PLOT_COLORS["reconstructed_edge"],
        linewidth=1.4,
        alpha=0.75,
    )
    _finalize_axis(ax, [orig_coords, recon_coords], bounds=axis_bounds)


def _draw_mask_state_overview(
    ax,
    orig_coords,
    masked_ids,
    *,
    title,
    unmasked_color="#1f6ff6",
    masked_color="#b5b5b5",
    bridge_color="#cfcfcf",
    unmasked_size=52,
    masked_size=48,
    edge_width=1.4,
    bridge_width=1.0,
    axis_bounds=None,
):
    _setup_axis(ax, title)
    total_joints = orig_coords[0].shape[0]
    all_indices = np.arange(total_joints)
    masked_ids = np.asarray(masked_ids, dtype=int)
    if masked_ids.size == 0:
        masked_ids = np.array([], dtype=int)
    unmasked_ids = np.setdiff1d(all_indices, masked_ids, assume_unique=True)
    masked_set = set(masked_ids.tolist())
    unmasked_set = set(unmasked_ids.tolist())

    if unmasked_ids.size > 0:
        _scatter_points(
            ax,
            orig_coords,
            unmasked_ids,
            color=unmasked_color,
            size=unmasked_size,
            alpha=0.95,
            marker="o",
        )
    if masked_ids.size > 0:
        _scatter_points(
            ax,
            orig_coords,
            masked_ids,
            color=masked_color,
            size=masked_size,
            alpha=0.35,
            marker="o",
            edgecolor=None,
            linewidth=0.0,
        )

    if unmasked_set:
        _draw_edges(
            ax,
            orig_coords,
            predicate=lambda i, j: i in unmasked_set and j in unmasked_set,
            color=unmasked_color,
            linewidth=edge_width,
            alpha=0.85,
        )
    if masked_set:
        _draw_edges(
            ax,
            orig_coords,
            predicate=lambda i, j: i in masked_set and j in masked_set,
            color=masked_color,
            linewidth=edge_width,
            alpha=0.3,
        )
    if masked_set and unmasked_set:
        _draw_edges(
            ax,
            orig_coords,
            predicate=lambda i, j: (i in masked_set) ^ (j in masked_set),
            color=bridge_color,
            linewidth=bridge_width,
            alpha=0.35,
        )

    _finalize_axis(ax, [orig_coords], bounds=axis_bounds)


def _draw_overlay_view(
    ax,
    orig_coords,
    mask_coords,
    mask_coords_raw,
    recon_coords,
    masked_ids,
    unmasked_ids,
    *,
    title,
    base_size=22,
    observed_size=55,
    token_size=32,
    masked_size=58,
    recon_mask_size=68,
    recon_visible_size=50,
    axis_bounds=None,
):
    _setup_axis(ax, title)
    _scatter_points(
        ax,
        orig_coords,
        color=PLOT_COLORS["original_cloud"],
        size=base_size,
        alpha=0.3,
    )
    if unmasked_ids.size > 0:
        _scatter_points(
            ax,
            mask_coords,
            unmasked_ids,
            color=PLOT_COLORS["visible_joint"],
            size=observed_size,
            alpha=0.9,
        )
        _scatter_points(
            ax,
            recon_coords,
            unmasked_ids,
            color=PLOT_COLORS["reconstructed_joint"],
            size=recon_visible_size,
            alpha=0.8,
        )
    if masked_ids.size > 0:
        _scatter_points(
            ax,
            mask_coords_raw,
            masked_ids,
            color=PLOT_COLORS["masked_placeholder"],
            size=token_size,
            marker="x",
            alpha=0.25,
            linewidth=0.6,
        )
        _scatter_points(
            ax,
            orig_coords,
            masked_ids,
            color=PLOT_COLORS["masked_joint"],
            size=masked_size,
            marker="o",
            alpha=0.35,
            edgecolor=None,
            linewidth=0.0,
        )
        _scatter_points(
            ax,
            recon_coords,
            masked_ids,
            color=PLOT_COLORS["masked_joint"],
            size=recon_mask_size,
            marker="^",
            alpha=0.42,
            edgecolor=None,
            linewidth=0.0,
        )
        x, y, z = orig_coords
        xr, yr, zr = recon_coords
        for idx in masked_ids:
            ax.plot(
                [x[idx], xr[idx]],
                [y[idx], yr[idx]],
                [z[idx], zr[idx]],
                color="#b3b3b3",
                linestyle="--",
                linewidth=0.7,
                alpha=0.4,
            )
    masked_set = set(masked_ids.tolist())
    unmasked_set = set(unmasked_ids.tolist())
    _draw_edges(
        ax,
        orig_coords,
        color=PLOT_COLORS["original_edge"],
        linewidth=1.1,
        alpha=0.35,
    )
    _draw_edges(
        ax,
        mask_coords,
        predicate=lambda i, j: i in unmasked_set and j in unmasked_set,
        color=PLOT_COLORS["visible_edge"],
        linewidth=2.0,
        alpha=0.8,
    )
    _draw_edges(
        ax,
        orig_coords,
        predicate=lambda i, j: i in masked_set and j in masked_set,
        color=PLOT_COLORS["masked_edge"],
        linewidth=0.9,
        alpha=0.28,
    )
    _draw_edges(
        ax,
        recon_coords,
        color=PLOT_COLORS["reconstructed_edge"],
        linewidth=1.8,
        alpha=0.75,
    )
    _finalize_axis(
        ax,
        [
            orig_coords,
            mask_coords,
            recon_coords,
            _subset_coords(mask_coords_raw, masked_ids),
        ],
        bounds=axis_bounds,
    )


def plot_skeleton_visualization(original_data, masked_data, mask_indices, save_path, title="Skeleton Visualization"):
    """スケルトンの可視化（マスク位置表示版）"""
    original_patch = original_data[0]
    masked_patch = masked_data[0]
    masked_ids = get_frame_mask_numpy(mask_indices, sample_idx=0, frame_idx=0)
    masked_ids = np.sort(masked_ids)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    orig_frame = to_numpy_array(original_patch[0])
    mask_frame = to_numpy_array(masked_patch[0])
    orig_x, orig_y, orig_z = to_camera_coords(orig_frame)
    mask_x_raw, mask_y_raw, mask_z_raw = to_camera_coords(mask_frame)

    mask_x = mask_x_raw.copy()
    mask_y = mask_y_raw.copy()
    mask_z = mask_z_raw.copy()
    if masked_ids.size > 0:
        mask_x[masked_ids] = orig_x[masked_ids]
        mask_y[masked_ids] = orig_y[masked_ids]
        mask_z[masked_ids] = orig_z[masked_ids]

    joint_idx = np.arange(orig_frame.shape[0])
    unmasked_mask = ~np.isin(joint_idx, masked_ids)
    unmasked_indices = joint_idx[unmasked_mask]
    unmasked_set = set(unmasked_indices.tolist())

    ax.scatter(
        orig_x,
        orig_y,
        orig_z,
        color=PLOT_COLORS["original_cloud"],
        s=22,
        alpha=0.3,
        depthshade=False,
    )

    if unmasked_indices.size > 0:
        ax.scatter(
            mask_x[unmasked_indices],
            mask_y[unmasked_indices],
            mask_z[unmasked_indices],
            color=PLOT_COLORS["visible_joint"],
            s=55,
            alpha=0.85,
            depthshade=False,
        )

    if masked_ids.size > 0:
        ax.scatter(
            orig_x[masked_ids],
            orig_y[masked_ids],
            orig_z[masked_ids],
            color=PLOT_COLORS["masked_joint"],
            s=70,
            marker="o",
            edgecolor=None,
            linewidths=0.0,
            alpha=0.35,
            depthshade=False,
        )

    for edge in SKELETON_EDGES:
        if edge[0] < orig_frame.shape[0] and edge[1] < orig_frame.shape[0]:
            ax.plot(
                [orig_x[edge[0]], orig_x[edge[1]]],
                [orig_y[edge[0]], orig_y[edge[1]]],
                [orig_z[edge[0]], orig_z[edge[1]]],
                color=PLOT_COLORS["original_edge"],
                linewidth=1.2,
                alpha=0.4,
            )
            if edge[0] in unmasked_set and edge[1] in unmasked_set:
                ax.plot(
                    [mask_x[edge[0]], mask_x[edge[1]]],
                    [mask_y[edge[0]], mask_y[edge[1]]],
                    [mask_z[edge[0]], mask_z[edge[1]]],
                    color=PLOT_COLORS["visible_edge"],
                    linewidth=2.0,
                    alpha=0.75,
                )

    ax.set_title(title)
    ax.set_facecolor("#fbfbfb")
    ax.grid(False)

    set_axes_equal(
        ax,
        [orig_x, mask_x],
        [orig_y, mask_y],
        [orig_z, mask_z],
        margin=0.08,
    )
    set_camera_view(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstruction_comparison(original_data, masked_data, reconstructed_data, mask_indices, save_path, overlay=False):
    """再構成結果の比較可視化"""
    orig_sequence = to_numpy_array(original_data[0])
    mask_sequence = to_numpy_array(masked_data[0])
    recon_sequence = to_numpy_array(reconstructed_data[0])

    masked_ids = get_frame_mask_numpy(mask_indices, sample_idx=0, frame_idx=0)
    masked_ids = np.sort(masked_ids)

    orig_frame = orig_sequence[0]
    mask_frame = mask_sequence[0]
    recon_frame = recon_sequence[0]

    orig_coords = tuple(to_camera_coords(orig_frame))
    mask_coords_raw = tuple(to_camera_coords(mask_frame))
    recon_coords = tuple(to_camera_coords(recon_frame))
    mask_coords = _mask_observed_coords(mask_coords_raw, orig_coords, masked_ids)

    joint_idx = np.arange(orig_frame.shape[0])
    unmasked_ids = np.setdiff1d(joint_idx, masked_ids)

    axis_bounds = compute_coord_axis_bounds(
        [orig_coords, mask_coords, mask_coords_raw, recon_coords],
        margin=0.08,
    )

    if overlay:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        _draw_overlay_view(
            ax,
            orig_coords,
            mask_coords,
            mask_coords_raw,
            recon_coords,
            masked_ids,
            unmasked_ids,
            title="Skeleton Reconstruction Comparison (Overlay)",
            axis_bounds=axis_bounds,
        )
        plt.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig = plt.figure(figsize=(22, 5))

    ax1 = fig.add_subplot(141, projection="3d")
    _draw_original_view(ax1, orig_coords, title="Original", axis_bounds=axis_bounds)

    ax2 = fig.add_subplot(142, projection="3d")
    _draw_mask_view(
        ax2,
        orig_coords,
        mask_coords,
        mask_coords_raw,
        masked_ids,
        unmasked_ids,
        title="Masked (annotated)",
        axis_bounds=axis_bounds,
    )

    ax3 = fig.add_subplot(143, projection="3d")
    _draw_reconstructed_plain_view(ax3, recon_coords, title="Reconstructed", axis_bounds=axis_bounds)

    ax4 = fig.add_subplot(144, projection="3d")
    _draw_reconstructed_error_view(
        ax4,
        orig_coords,
        recon_coords,
        masked_ids,
        unmasked_ids,
        title="Reconstructed (error view)",
        axis_bounds=axis_bounds,
    )

    plt.tight_layout(w_pad=1.2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sequence_reconstruction_comparison(original_data, masked_data, reconstructed_data, mask_indices, save_path, overlay=False):
    """9フレームシーケンス全体の再構成結果比較可視化"""
    orig_sequence = to_numpy_array(original_data[0])
    mask_sequence = to_numpy_array(masked_data[0])
    recon_sequence = to_numpy_array(reconstructed_data[0])

    seq_len = orig_sequence.shape[0]
    joint_idx = np.arange(orig_sequence.shape[1])

    masked_indices_per_frame = []
    unmasked_indices_per_frame = []
    orig_coords_list = []
    mask_coords_raw_list = []
    mask_coords_list = []
    recon_coords_list = []

    for frame_idx in range(seq_len):
        masked_ids = get_frame_mask_numpy(mask_indices, sample_idx=0, frame_idx=frame_idx)
        masked_ids = np.sort(masked_ids)
        masked_indices_per_frame.append(masked_ids)
        unmasked_ids = np.setdiff1d(joint_idx, masked_ids)
        unmasked_indices_per_frame.append(unmasked_ids)

        orig_coords = tuple(to_camera_coords(orig_sequence[frame_idx]))
        mask_coords_raw = tuple(to_camera_coords(mask_sequence[frame_idx]))
        recon_coords = tuple(to_camera_coords(recon_sequence[frame_idx]))
        mask_coords = _mask_observed_coords(mask_coords_raw, orig_coords, masked_ids)

        orig_coords_list.append(orig_coords)
        mask_coords_raw_list.append(mask_coords_raw)
        mask_coords_list.append(mask_coords)
        recon_coords_list.append(recon_coords)

    global_bounds = compute_coord_axis_bounds(
        orig_coords_list + mask_coords_list + mask_coords_raw_list + recon_coords_list,
        margin=0.08,
    )

    if overlay:
        fig = plt.figure(figsize=(4 * seq_len, 4.5))
        for frame_idx in range(seq_len):
            ax = fig.add_subplot(1, seq_len, frame_idx + 1, projection="3d")
            _draw_mask_view(
                ax,
                orig_coords_list[frame_idx],
                mask_coords_list[frame_idx],
                mask_coords_raw_list[frame_idx],
                masked_indices_per_frame[frame_idx],
                unmasked_indices_per_frame[frame_idx],
                title=f"Mask info F{frame_idx + 1}",
                visible_size=45,
                masked_size=58,
                token_size=32,
                visible_edge_width=1.5,
                masked_edge_width=0.9,
                axis_bounds=global_bounds,
            )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig = plt.figure(figsize=(4 * seq_len, 12))

    for frame_idx in range(seq_len):
        base_index = frame_idx + 1

        ax_orig = fig.add_subplot(3, seq_len, base_index, projection="3d")
        _draw_original_view(
            ax_orig,
            orig_coords_list[frame_idx],
            title=f"Original F{frame_idx + 1}",
            point_size=30,
            edge_width=1.1,
            alpha=0.9,
            axis_bounds=global_bounds,
        )

        ax_recon = fig.add_subplot(3, seq_len, base_index + seq_len, projection="3d")
        _draw_reconstructed_plain_view(
            ax_recon,
            recon_coords_list[frame_idx],
            title=f"Reconstruct F{frame_idx + 1}",
            point_size=40,
            edge_width=1.4,
            alpha=0.9,
            axis_bounds=global_bounds,
        )

        ax_compare = fig.add_subplot(3, seq_len, base_index + 2 * seq_len, projection="3d")
        _draw_reconstructed_error_view(
            ax_compare,
            orig_coords_list[frame_idx],
            recon_coords_list[frame_idx],
            masked_indices_per_frame[frame_idx],
            unmasked_indices_per_frame[frame_idx],
            title=f"Comparison F{frame_idx + 1}",
            visible_size=42,
            masked_size=52,
            dashed_alpha=0.65,
            dashed_width=0.9,
            orig_alpha=0.25,
            axis_bounds=global_bounds,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sequence_mask_overview(original_data, mask_indices, save_path, sample_idx=0):
    """各フレームのマスク状態を可視化（未マスク:青, マスク:濃いグレー）。"""
    sample_original = original_data[sample_idx]
    orig_sequence = to_numpy_array(sample_original)
    seq_len = orig_sequence.shape[0]

    coords_list = []
    masked_ids_list = []
    for frame_idx in range(seq_len):
        coords = tuple(to_camera_coords(orig_sequence[frame_idx]))
        masked_ids = get_frame_mask_numpy(mask_indices, sample_idx=sample_idx, frame_idx=frame_idx)
        coords_list.append(coords)
        masked_ids_list.append(masked_ids)

    global_bounds = compute_coord_axis_bounds(coords_list, margin=0.08)

    fig = plt.figure(figsize=(4 * seq_len, 4.2))
    for frame_idx in range(seq_len):
        ax = fig.add_subplot(1, seq_len, frame_idx + 1, projection="3d")
        _draw_mask_state_overview(
            ax,
            coords_list[frame_idx],
            masked_ids_list[frame_idx],
            title=f"Mask F{frame_idx + 1}",
            axis_bounds=global_bounds,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def calculate_joint_errors(original, reconstructed, mask_indices, device, sample_idx=0):
    """
    各関節の物理的距離誤差（メートル）を計算（1サンプル単位）
    """
    batch_size, seq_len, num_joints, coords = original.shape

    original_sample = original[sample_idx]
    reconstructed_sample = reconstructed[sample_idx]
    sample_mask_indices = mask_indices[sample_idx]
    masked_union = collapse_mask_indices(sample_mask_indices)
    masked_joints_set = set(masked_union.tolist())

    joint_errors = []
    masked_joint_errors = []
    unmasked_joint_errors = []

    for joint_idx in range(num_joints):
        joint_original = original_sample[:, joint_idx, :]
        joint_reconstructed = reconstructed_sample[:, joint_idx, :]

        joint_diff = joint_original - joint_reconstructed
        joint_distances = torch.norm(joint_diff, dim=-1)
        joint_avg_error = joint_distances.mean().item()

        joint_errors.append(joint_avg_error)

        if joint_idx in masked_joints_set:
            masked_joint_errors.append(joint_avg_error)
        else:
            unmasked_joint_errors.append(joint_avg_error)

    return joint_errors, masked_joint_errors, unmasked_joint_errors


def calculate_masked_unmasked_batch_errors(original, reconstructed, mask_indices):
    """
    バッチ全体でのマスク/非マスク平均誤差をフレーム単位で集計（1関節あたり）。
    物理的距離（メートル）で計算。
    """
    B, T, V, _ = original.shape
    distances = torch.norm(original - reconstructed, dim=-1)
    per_joint_masked_vals = [[] for _ in range(V)]
    per_joint_unmasked_vals = [[] for _ in range(V)]
    total_masked_instances = 0
    total_unmasked_instances = 0
    for b in range(B):
        for t in range(T):
            if mask_indices is None:
                raw_mask = torch.empty(0, dtype=torch.long, device=distances.device)
            else:
                try:
                    raw_mask = mask_indices[b][t]
                except (TypeError, IndexError):
                    raw_mask = get_frame_mask_indices(mask_indices, sample_idx=b, frame_idx=t)
            frame_mask = _ensure_long_tensor(raw_mask, device=distances.device)
            if frame_mask.numel() > 0:
                frame_mask = torch.unique(frame_mask)

            mask_flags = torch.zeros(V, dtype=torch.bool, device=distances.device)
            if frame_mask.numel() > 0:
                mask_flags[frame_mask] = True

            frame_dist = distances[b, t].detach().cpu().numpy()
            for v in range(V):
                val = float(frame_dist[v])
                if mask_flags[v]:
                    per_joint_masked_vals[v].append(val)
                    total_masked_instances += 1
                else:
                    per_joint_unmasked_vals[v].append(val)
                    total_unmasked_instances += 1
    per_joint_masked_mean = np.array([np.mean(x) if len(x) > 0 else np.nan for x in per_joint_masked_vals])
    per_joint_unmasked_mean = np.array([np.mean(x) if len(x) > 0 else np.nan for x in per_joint_unmasked_vals])
    per_joint_masked_count = np.array([len(x) for x in per_joint_masked_vals])
    per_joint_unmasked_count = np.array([len(x) for x in per_joint_unmasked_vals])
    micro_masked_mean = float(np.mean([v for lst in per_joint_masked_vals for v in lst])) if total_masked_instances > 0 else 0.0
    micro_unmasked_mean = float(np.mean([v for lst in per_joint_unmasked_vals for v in lst])) if total_unmasked_instances > 0 else 0.0
    macro_masked_mean = float(np.nanmean(per_joint_masked_mean)) if np.any(~np.isnan(per_joint_masked_mean)) else 0.0
    macro_unmasked_mean = float(np.nanmean(per_joint_unmasked_mean)) if np.any(~np.isnan(per_joint_unmasked_mean)) else 0.0
    avg_masked_per_sample = total_masked_instances / float(B * T)
    mask_rate = total_masked_instances / float(B * T * V)
    return {
        "micro_masked_mean": micro_masked_mean,
        "micro_unmasked_mean": micro_unmasked_mean,
        "macro_masked_mean": macro_masked_mean,
        "macro_unmasked_mean": macro_unmasked_mean,
        "per_joint_masked_mean": per_joint_masked_mean,
        "per_joint_unmasked_mean": per_joint_unmasked_mean,
        "per_joint_masked_count": per_joint_masked_count,
        "per_joint_unmasked_count": per_joint_unmasked_count,
        "total_masked_instances": total_masked_instances,
        "total_unmasked_instances": total_unmasked_instances,
        "avg_masked_per_sample": avg_masked_per_sample,
        "mask_rate": mask_rate,
    }


