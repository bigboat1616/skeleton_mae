import logging
import yaml
import json
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader

from graphmae.utils import build_args, load_best_configs
from graphmae.models.stgcn import ST_GCN_18

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


SKELETON_EDGES = [
    (0, 1), (1, 2),  # 頭部
    (2, 3), (3, 4), (4, 5), (5, 6),  # 右腕
    (2, 7), (7, 8), (8, 9), (9, 10),  # 左腕
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # 脊椎
    (15, 16), (16, 17), (17, 18),  # 右足
    (15, 19), (19, 20), (20, 21),  # 左足
]

PLOT_COLORS = {
    'original_cloud': '#b3cde3',
    'original_edge': '#9ecae1',
    'visible_joint': '#1f77b4',
    'visible_edge': '#1f77b4',
    'masked_joint': '#ff7f0e',
    'masked_placeholder': '#d62728',
    'reconstructed_joint': '#2ca02c',
    'reconstructed_edge': '#41ab5d',
}

DEFAULT_VIEW = {
    'elev': 180.0,
    'azim':180.0,
}


def to_numpy_indices(indices):
    """Convert masked joint indices to a numpy array."""
    if indices is None:
        return np.array([], dtype=int)
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(int)
    if isinstance(indices, (list, tuple, np.ndarray)):
        return np.asarray(indices, dtype=int)
    return np.array([int(indices)], dtype=int)


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


def mask_skeleton_joints(data, mask_ratio=0.15, mask_token=0.0, mask_indices=None):
    """
    スケルトンデータの関節をマスクする（座標空間・特徴空間共通）
    
    Args:
        data: [batch_size, seq_len, num_joints, C] のデータ（C=3 for 座標, C=feature_dim for 特徴）
        mask_ratio: マスクする関節の割合（mask_indicesがNoneの場合のみ使用）
        mask_token: マスクされた関節に設定する値（スカラー or テンソル）
        mask_indices: マスクされた関節のインデックス（指定された場合は使用）
    
    Returns:
        masked_data: マスクされたデータ
        mask_indices: マスクされた関節のインデックス
    """
    batch_size, seq_len, num_joints, channels = data.shape
    masked_data = data.clone()
    device = data.device
    
    # マスクインデックスが指定されていない場合は生成
    if mask_indices is None:
        mask_indices = []
        for b in range(batch_size):
            # マスクする関節数を計算
            num_masked = int(num_joints * mask_ratio)
            
            # ランダムにマスクする関節を選択（デバイス一致）
            masked_joints = torch.randperm(num_joints, device=device)[:num_masked]
            mask_indices.append(masked_joints)
    
    # バッチごとにマスク
    for b in range(batch_size):
        masked_joints = mask_indices[b]
        if len(masked_joints) > 0:
            # マスクトークンがテンソルの場合（特徴空間）とスカラーの場合（座標空間）を処理
            if isinstance(mask_token, torch.Tensor):
                # 特徴空間: マスクトークンをブロードキャスト
                if mask_token.dim() == 4:  # [1, 1, 1, feature_dim]
                    masked_data[b, :, masked_joints, :] = mask_token.squeeze(0).squeeze(0).squeeze(0)
                else:  # [feature_dim]
                    masked_data[b, :, masked_joints, :] = mask_token
            else:
                # 座標空間: スカラー値
                masked_data[b, :, masked_joints, :] = mask_token
    
    return masked_data, mask_indices


def plot_skeleton_visualization(original_data, masked_data, mask_indices, save_path, title="Skeleton Visualization"):
    """スケルトンの可視化（マスク位置表示版）"""
    original_patch = original_data[0]
    masked_patch = masked_data[0]
    masked_ids = to_numpy_indices(mask_indices[0])
    masked_ids = np.sort(masked_ids)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 座標変換（カメラ座標系: X下+, Y左+, Z奥+）
    orig_frame = to_numpy_array(original_patch[0])
    mask_frame = to_numpy_array(masked_patch[0])
    orig_x, orig_y, orig_z = to_camera_coords(orig_frame)
    mask_x_raw, mask_y_raw, mask_z_raw = to_camera_coords(mask_frame)

    # マスクされた関節は元の位置で表示する
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

    legend_entries = []

    # 1. 元のスケルトン全体（透明度を下げた薄い色で基準表示）
    handle = ax.scatter(
        orig_x, orig_y, orig_z,
        color=PLOT_COLORS['original_cloud'],
        s=22,
        alpha=0.3,
        label='Original (reference)',
        depthshade=False,
    )
    _add_legend_entry(legend_entries, handle, 'Original (reference)')

    # 2. モデルに入力される可視ジョイント
    if unmasked_indices.size > 0:
        handle = ax.scatter(
            mask_x[unmasked_indices],
            mask_y[unmasked_indices],
            mask_z[unmasked_indices],
            color=PLOT_COLORS['visible_joint'],
            s=55,
            alpha=0.85,
            label='Observed joints',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Observed joints')

    # 3. マスクされたジョイント（元位置を強調）
    if masked_ids.size > 0:
        handle = ax.scatter(
            orig_x[masked_ids],
            orig_y[masked_ids],
            orig_z[masked_ids],
            color=PLOT_COLORS['masked_joint'],
            s=90,
            marker='o',
            edgecolor='k',
            linewidths=0.6,
            alpha=0.95,
            label='Masked joints',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Masked joints')

    # 骨格接続をプロット
    for edge in SKELETON_EDGES:
        if edge[0] < orig_frame.shape[0] and edge[1] < orig_frame.shape[0]:
            ax.plot(
                [orig_x[edge[0]], orig_x[edge[1]]],
                [orig_y[edge[0]], orig_y[edge[1]]],
                [orig_z[edge[0]], orig_z[edge[1]]],
                color=PLOT_COLORS['original_edge'],
                linewidth=1.2,
                alpha=0.4,
            )
            if edge[0] in unmasked_set and edge[1] in unmasked_set:
                ax.plot(
                    [mask_x[edge[0]], mask_x[edge[1]]],
                    [mask_y[edge[0]], mask_y[edge[1]]],
                    [mask_z[edge[0]], mask_z[edge[1]]],
                    color=PLOT_COLORS['visible_edge'],
                    linewidth=2.0,
                    alpha=0.75,
                )
    
    ax.set_xlabel('X (down +)')
    ax.set_ylabel('Y (left +)')
    ax.set_zlabel('Z (forward +)')
    ax.set_title(title)
    ax.set_facecolor('#fbfbfb')
    ax.grid(False)

    set_axes_equal(
        ax,
        [orig_x, mask_x],
        [orig_y, mask_y],
        [orig_z, mask_z],
        margin=0.08,
    )
    set_camera_view(ax)
    handles, labels = _unique_legend_entries(legend_entries)
    ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")
    print(f"Masked joints: {masked_ids.tolist()}")



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


def set_axes_equal(ax, x_arrays, y_arrays, z_arrays, margin=0.05):
    """Set equal scale on all axes while keeping a configurable margin."""
    x_vals = _collect_axis_values(x_arrays)
    y_vals = _collect_axis_values(y_arrays)
    z_vals = _collect_axis_values(z_arrays)

    def _limits(values):
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        span = max(max_val - min_val, 1e-6)
        pad = span * margin
        return min_val - pad, max_val + pad, max(span, 1e-6)

    x_min, x_max, x_span = _limits(x_vals)
    y_min, y_max, y_span = _limits(y_vals)
    z_min, z_max, z_span = _limits(z_vals)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_span, y_span, z_span))


def set_camera_view(ax, elev=None, azim=None):
    """Apply a consistent front-facing camera view without perspective distortion."""
    elev = DEFAULT_VIEW['elev'] if elev is None else elev
    azim = DEFAULT_VIEW['azim'] if azim is None else azim
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type('ortho')
    except AttributeError:
        # Older matplotlib versions may not support set_proj_type
        pass


def compute_reconstruction_loss(original, reconstructed, mask_indices, loss_type='mse', beta=2.0):
    """
    再構成損失を計算する（MSEまたはRCE）
    
    Args:
        original: 元のスケルトンデータ [batch_size, seq_len, num_joints, 3]
        reconstructed: 再構成されたスケルトンデータ [batch_size, seq_len, num_joints, 3]
        mask_indices: マスクされた関節のインデックス
        loss_type: 損失関数のタイプ ('mse', 'l1', 'rce')
        beta: RCEの重み付けパラメータ（β ≥ 1）
    
    Returns:
        loss: 損失値
        loss_dict: 損失の詳細情報
    """
    batch_size = original.shape[0]
    device = original.device
    
    # 損失関数の選択
    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction='none')
    elif loss_type == 'rce':
        # RCE (Re-weighted Cosine Error) の実装
        def rce_loss(x, y, beta=beta):
            """
            Re-weighted Cosine Error
            LRCE = Σ(1/|V| - (xT·y)/(|V|×||x||×||y||))^β
            """
            batch_size, seq_len, num_joints, coords = x.shape
            
            # 各関節のコサイン類似度を計算
            x_flat = x.view(batch_size, seq_len, num_joints, coords)  # [batch_size, seq_len, num_joints, 3]
            y_flat = y.view(batch_size, seq_len, num_joints, coords)  # [batch_size, seq_len, num_joints, 3]
            
            # 各関節のノルムを計算
            x_norm = torch.norm(x_flat, dim=3, keepdim=True)  # [batch_size, seq_len, num_joints, 1]
            y_norm = torch.norm(y_flat, dim=3, keepdim=True)  # [batch_size, seq_len, num_joints, 1]
            
            # コサイン類似度
            cosine_sim = torch.sum(x_flat * y_flat, dim=3, keepdim=True) / (x_norm * y_norm + 1e-8)
            
            # RCE計算: (1 - cosine_sim)^β
            rce = torch.pow(1 - cosine_sim, beta)
            
            # 形状を [batch_size, seq_len, num_joints, 3] に合わせる
            rce_expanded = rce.expand(batch_size, seq_len, num_joints, coords)
            
            return rce_expanded
        
        loss_fn = rce_loss
    else:
        loss_fn = nn.MSELoss(reduction='none')
    
    # 全関節の損失を計算
    all_losses = loss_fn(reconstructed, original)  # [batch_size, seq_len, num_joints, 3]
    
    # バッチごとに処理
    masked_losses = []
    unmasked_losses = []
    
    for b in range(batch_size):
        masked_joints = mask_indices[b]
        if isinstance(masked_joints, torch.Tensor):
            masked_joints = masked_joints.to(device)
        else:
            masked_joints = torch.tensor(masked_joints, device=device)
        
        # 全関節のインデックス
        all_joints = torch.arange(original.shape[2], device=device)
        unmasked_joints = all_joints[~torch.isin(all_joints, masked_joints)]
        
        # マスクされた関節の損失
        if len(masked_joints) > 0:
            masked_loss = all_losses[b, :, masked_joints, :].mean()
            masked_losses.append(masked_loss)
        
        # マスクされていない関節の損失
        if len(unmasked_joints) > 0:
            unmasked_loss = all_losses[b, :, unmasked_joints, :].mean()
            unmasked_losses.append(unmasked_loss)
    
    # 平均損失の計算
    masked_avg_loss = torch.stack(masked_losses).mean() if masked_losses else torch.tensor(0.0, device=device)
    unmasked_avg_loss = torch.stack(unmasked_losses).mean() if unmasked_losses else torch.tensor(0.0, device=device)
    
    # 総損失
    total_loss = masked_avg_loss + unmasked_avg_loss
    
    # 統計情報
    num_masked_joints = np.mean([len(mask_idx) for mask_idx in mask_indices]) if mask_indices else 0
    num_unmasked_joints = original.shape[2] - num_masked_joints
    
    # 関節あたりの損失（正規化）
    total_loss_per_joint = total_loss.item() / original.shape[2]
    masked_loss_per_joint = masked_avg_loss.item() / num_masked_joints if num_masked_joints > 0 else 0.0
    unmasked_loss_per_joint = unmasked_avg_loss.item() / num_unmasked_joints if num_unmasked_joints > 0 else 0.0
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'total_loss_per_joint': total_loss_per_joint,
        'masked_joints_loss_mean': masked_avg_loss.item(),
        'masked_joints_loss_std': 0.0,  # 簡略化
        'masked_joints_loss_per_joint': masked_loss_per_joint,
        'unmasked_joints_loss_mean': unmasked_avg_loss.item(),
        'unmasked_joints_loss_std': 0.0,  # 簡略化
        'unmasked_joints_loss_per_joint': unmasked_loss_per_joint,
        'num_masked_joints': num_masked_joints,
        'num_unmasked_joints': num_unmasked_joints,
    }
    
    return total_loss, loss_dict




class STGCN18Reconstructor(nn.Module):
    """ST-GCN-18を用いた座標再構成モデル"""

    def __init__(self, in_channels=3, out_channels=3, feature_dim=256):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_channels = out_channels

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, in_channels))

        graph_cfg = {
            'layout': 'jta_3dp_row',
            'strategy': 'distance',
            'max_hop': 1,
            'dilation': 1
        }

        self.encoder = ST_GCN_18(
            in_channels=in_channels,
            feature_dim=feature_dim,
            graph_cfg=graph_cfg,
            edge_importance_weighting=True,
            data_bn=True
        )

        self.coord_decoder = nn.Linear(feature_dim, out_channels)

    def forward(self, x, return_features=False):
        batch_size, seq_len, num_joints, _ = x.shape

        encoded, _ = self.encoder.extract_feature(x.permute(0, 3, 1, 2))
        encoded = encoded.permute(0, 2, 3, 1)

        decoded = self.coord_decoder(encoded.reshape(-1, self.feature_dim))
        reconstructed = decoded.view(batch_size, seq_len, num_joints, self.out_channels)

        if return_features:
            return reconstructed, encoded

        return reconstructed



def calculate_joint_errors(original, reconstructed, mask_indices, device, sample_idx=0):
    """
    各関節の物理的距離誤差（メートル）を計算（1サンプル単位）
    
    Args:
        original: [batch_size, seq_len, num_joints, 3] 元の座標
        reconstructed: [batch_size, seq_len, num_joints, 3] 再構成座標
        mask_indices: マスクされた関節のインデックス
        device: 使用デバイス
        sample_idx: 表示するサンプルのインデックス（デフォルト: 0）
    
    Returns:
        joint_errors: [num_joints] 各関節の平均誤差（メートル）
        masked_joint_errors: [num_masked_joints] マスクされた関節の誤差
        unmasked_joint_errors: [num_unmasked_joints] マスクされていない関節の誤差
    """
    batch_size, seq_len, num_joints, coords = original.shape
    
    # 指定されたサンプルのみを使用
    original_sample = original[sample_idx]  # [seq_len, num_joints, 3]
    reconstructed_sample = reconstructed[sample_idx]  # [seq_len, num_joints, 3]
    sample_mask_indices = mask_indices[sample_idx]  # そのサンプルのマスクインデックス
    
    # 全関節の誤差を計算
    joint_errors = []
    masked_joint_errors = []
    unmasked_joint_errors = []
    
    # マスクされた関節のセットを作成
    masked_joints_set = set()
    for masked_joint in sample_mask_indices:
        if isinstance(masked_joint, torch.Tensor):
            masked_joints_set.add(masked_joint.item())
        else:
            masked_joints_set.add(int(masked_joint))
    
    for joint_idx in range(num_joints):
        # 関節ごとの誤差を計算
        joint_original = original_sample[:, joint_idx, :]  # [seq_len, 3]
        joint_reconstructed = reconstructed_sample[:, joint_idx, :]  # [seq_len, 3]
        
        # ユークリッド距離を計算
        joint_diff = joint_original - joint_reconstructed  # [seq_len, 3]
        joint_distances = torch.norm(joint_diff, dim=-1)  # [seq_len]
        joint_avg_error = joint_distances.mean().item()  # 平均誤差（メートル）
        
        joint_errors.append(joint_avg_error)
        
        # マスクされた関節かどうかを判定
        if joint_idx in masked_joints_set:
            masked_joint_errors.append(joint_avg_error)
        else:
            unmasked_joint_errors.append(joint_avg_error)
    
    return joint_errors, masked_joint_errors, unmasked_joint_errors



def calculate_masked_unmasked_batch_errors(original, reconstructed, mask_indices):
    """
    バッチ全体でのマスク/非マスク平均誤差を直感的に集計（1関節あたり）。
    物理的距離（メートル）で計算。
    - micro 平均: すべての関節-サンプルの実例で重み付け（頻度重み付き）
    - macro 平均: 関節ごとの平均を取ってからジョイントで平均（関節ごと等重み）
    さらに、関節ごとの平均誤差とサンプル数も返す。
    
    Args:
        original: [B, T, V, 3] 元の座標
        reconstructed: [B, T, V, 3] 再構成された座標
        mask_indices: list(len=B) of Tensor/list with masked joint indices per sample
    Returns:
        stats: dict with keys:
            - micro_masked_mean, micro_unmasked_mean
            - macro_masked_mean, macro_unmasked_mean
            - per_joint_masked_mean: np.ndarray shape [V] (NaN if該当なし)
            - per_joint_unmasked_mean: np.ndarray shape [V]
            - per_joint_masked_count: np.ndarray shape [V]
            - per_joint_unmasked_count: np.ndarray shape [V]
            - total_masked_instances, total_unmasked_instances
            - avg_masked_per_sample, mask_rate
    """
    B, T, V, _ = original.shape
    # [B, T, V] - 物理的距離（メートル）
    distances = torch.norm(original - reconstructed, dim=-1)
    # 時系列平均 -> [B, V]
    per_sample_joint = distances.mean(dim=1).cpu().numpy()
    # マスク集合（各サンプルごとに set）
    masked_sets = []
    for b in range(B):
        ms = set(int(j.item()) if isinstance(j, torch.Tensor) else int(j) for j in mask_indices[b])
        masked_sets.append(ms)
    # 関節ごとに値を収集
    per_joint_masked_vals = [[] for _ in range(V)]
    per_joint_unmasked_vals = [[] for _ in range(V)]
    total_masked_instances = 0
    total_unmasked_instances = 0
    for b in range(B):
        for v in range(V):
            val = float(per_sample_joint[b, v])
            if v in masked_sets[b]:
                per_joint_masked_vals[v].append(val)
                total_masked_instances += 1
            else:
                per_joint_unmasked_vals[v].append(val)
                total_unmasked_instances += 1
    # per-joint mean と count
    per_joint_masked_mean = np.array([np.mean(x) if len(x) > 0 else np.nan for x in per_joint_masked_vals])
    per_joint_unmasked_mean = np.array([np.mean(x) if len(x) > 0 else np.nan for x in per_joint_unmasked_vals])
    per_joint_masked_count = np.array([len(x) for x in per_joint_masked_vals])
    per_joint_unmasked_count = np.array([len(x) for x in per_joint_unmasked_vals])
    # micro: 全実例で平均
    micro_masked_mean = float(np.mean([v for lst in per_joint_masked_vals for v in lst])) if total_masked_instances > 0 else 0.0
    micro_unmasked_mean = float(np.mean([v for lst in per_joint_unmasked_vals for v in lst])) if total_unmasked_instances > 0 else 0.0
    # macro: ジョイント平均
    macro_masked_mean = float(np.nanmean(per_joint_masked_mean)) if np.any(~np.isnan(per_joint_masked_mean)) else 0.0
    macro_unmasked_mean = float(np.nanmean(per_joint_unmasked_mean)) if np.any(~np.isnan(per_joint_unmasked_mean)) else 0.0
    # カバレッジ情報
    avg_masked_per_sample = total_masked_instances / float(B)
    mask_rate = total_masked_instances / float(B * V)
    return {
        'micro_masked_mean': micro_masked_mean,
        'micro_unmasked_mean': micro_unmasked_mean,
        'macro_masked_mean': macro_masked_mean,
        'macro_unmasked_mean': macro_unmasked_mean,
        'per_joint_masked_mean': per_joint_masked_mean,
        'per_joint_unmasked_mean': per_joint_unmasked_mean,
        'per_joint_masked_count': per_joint_masked_count,
        'per_joint_unmasked_count': per_joint_unmasked_count,
        'total_masked_instances': total_masked_instances,
        'total_unmasked_instances': total_unmasked_instances,
        'avg_masked_per_sample': avg_masked_per_sample,
        'mask_rate': mask_rate,
    }



def skeleton_pretrain(dataloader, device, mask_ratio=0.15, max_epochs=100, lr=0.001, weight_decay=0.01, save_dir=None, random_seed=42, loss_fn='mse', beta=2.0, feature_dim=256):
    """
    スケルトンデータに対するLinear再構成pretraining
    
    Args:
        dataloader: スケルトンデータのDataLoader
        device: 使用デバイス
        mask_ratio: マスクする関節の割合
        max_epochs: 最大エポック数
        lr: 学習率
        save_dir: 保存ディレクトリ
    
    Returns:
        model: 学習済みモデル
        training_history: 学習履歴
    """
    logging.info("Starting skeleton pretraining with Linear model...")
    
    # ランダムシードを固定（再現性確保）
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 再現性のための設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # モデル設定
    sample_batch = next(iter(dataloader))
    batch_size, seq_len, num_joints, coords = sample_batch.shape
    input_dim = seq_len * num_joints * coords  # 9 * 22 * 3 = 594
    
    # モデル初期化（ST-GCN-18ベース）
    model = STGCN18Reconstructor(in_channels=3, out_channels=3, feature_dim=feature_dim).to(device)
    print(f"Model initialized:")
    print(f"  - Model type: ST-GCN-18 Coordinate Reconstructor")
    print(f"  - Encoder: ST-GCN-18")
    print(f"  - Input channels: 3")
    print(f"  - Output channels: 3")
    print(f"  - Feature dimension: {feature_dim}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学習率スケジューラー（段階的減衰）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")
    
    # 学習履歴
    training_history = {
        'epoch': [],
        'loss': [],
        'all_joints_loss': [],
        'masked_joints_loss_mean': [],
        'masked_joints_loss_std': [],
        'unmasked_joints_loss_mean': [],
        'unmasked_joints_loss_std': []
    }
    
    # チェックポイント保存用ディレクトリ作成
    if save_dir:
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory created: {checkpoint_dir}")
    
    # 最良モデル追跡用
    best_loss = float('inf')
    best_model_state = None
    
    # 学習中の最後のバッチを保存（一貫性のため）
    last_batch = None
    last_mask_indices = None
    
    # 10エポックごとの再構成変化を追跡するための固定データ
    fixed_batch = sample_batch.clone().to(device)
    fixed_masked, fixed_mask_indices = mask_skeleton_joints(
        fixed_batch, mask_ratio=mask_ratio, mask_token=model.mask_token
    )
    print(f"Fixed batch for reconstruction tracking: {fixed_batch.shape}")
    print(f"Fixed masked joints count per sample: {[len(mask_idx) for mask_idx in fixed_mask_indices]}")
    print(f"Fixed masked joint indices: {[mask_idx.tolist() for mask_idx in fixed_mask_indices]}")
    print(f"Mask ratio: {mask_ratio:.1%} (expected: {int(22 * mask_ratio)} joints per sample)")
    
    # 再構成変化の履歴を保存
    reconstruction_history = []
    
    # 学習ループ
    model.train()
    epoch_iter = tqdm(range(max_epochs), desc="Training")

    for epoch in epoch_iter:
        epoch_losses = []
        epoch_masked_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # データをデバイスに移動
            batch = batch.to(device)
            
            # マスキング
            masked_batch, mask_indices = mask_skeleton_joints(
                batch, mask_ratio=mask_ratio, mask_token=model.mask_token
            )

            # マスクされた座標から再構成
            reconstructed = model(masked_batch)
            loss, loss_dict = compute_reconstruction_loss(
                batch,
                reconstructed,
                mask_indices,
                loss_type=loss_fn,
                beta=beta
            )
            
            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング（安定化）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 統計記録
            epoch_losses.append(loss.item())
            epoch_masked_losses.append(loss_dict['masked_joints_loss_mean'])
            
            # 最後のバッチを保存（一貫性のため）
            last_batch = batch.clone()
            last_mask_indices = mask_indices
        
        # エポック統計
        avg_loss = np.mean(epoch_losses)
        avg_masked_loss = np.mean(epoch_masked_losses)
        std_masked_loss = np.std(epoch_masked_losses)
        
        # 履歴更新
        training_history['epoch'].append(epoch)
        training_history['loss'].append(avg_loss)
        training_history['all_joints_loss'].append(avg_loss)  # 全ジョイント損失
        training_history['masked_joints_loss_mean'].append(avg_masked_loss)
        training_history['masked_joints_loss_std'].append(std_masked_loss)
        training_history['unmasked_joints_loss_mean'].append(loss_dict.get('unmasked_joints_loss_mean', 0.0))
        training_history['unmasked_joints_loss_std'].append(loss_dict.get('unmasked_joints_loss_std', 0.0))
        
        # 進捗表示
        epoch_iter.set_description(f"Epoch {epoch}: Loss={avg_loss:.6f}")
        
        # 最良モデルの更新
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            print(f"  🎯 New best model! Loss: {best_loss:.6f}")
        
       
        # 10エポックごとに再構成変化をプロット
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Generating reconstruction comparison at epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                fixed_reconstructed = model(fixed_masked)

            reconstruction_history.append({
                'epoch': epoch + 1,
                'original': fixed_batch.cpu(),
                'masked': fixed_masked.cpu(),
                'reconstructed': fixed_reconstructed.cpu(),
                'mask_indices': fixed_mask_indices
            })

            if save_dir:
                plot_path = f"{save_dir}/reconstruction_epoch_{epoch+1:03d}.png"
                plot_reconstruction_comparison(
                    fixed_batch,
                    fixed_masked,
                    fixed_reconstructed,
                    fixed_mask_indices,
                    plot_path
                )
                print(f"  ✅ Reconstruction plot saved: reconstruction_epoch_{epoch+1:03d}.png")

                sequence_plot_path = f"{save_dir}/sequence_reconstruction_epoch_{epoch+1:03d}.png"
                plot_sequence_reconstruction_comparison(
                    fixed_batch,
                    fixed_masked,
                    fixed_reconstructed,
                    fixed_mask_indices,
                    sequence_plot_path,
                    overlay=False
                )
                print(f"  ✅ Sequence reconstruction plot saved: sequence_reconstruction_epoch_{epoch+1:03d}.png")

            model.train()
        
        # 学習率スケジューラー更新
            scheduler.step()

            print(f"\n💾 Saving checkpoint at epoch {epoch+1}...")
            
            # 1. 全体の重み保存
            full_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'masked_joints_loss_mean': avg_masked_loss,
                'masked_joints_loss_std': std_masked_loss,
                'training_history': training_history,
                'config': {
                    'mask_ratio': mask_ratio,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'max_epochs': max_epochs
                }
            }
            torch.save(full_checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1:03d}.pth")
            print(f"  ✅ Full checkpoint saved: checkpoint_epoch_{epoch+1:03d}.pth")
            
            # 2. エンコーダーとcoord_to_featureの重みのみ保存
            encoder_state = {}
            for name, param in model.named_parameters():
                if 'encoder' in name or 'coord_to_feature' in name:
                    encoder_state[name] = param.data.clone()
            
            encoder_checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder_state,
                'loss': avg_loss,
                'config': {
                    'mask_ratio': mask_ratio,
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            }
            torch.save(encoder_checkpoint, f"{checkpoint_dir}/encoder_epoch_{epoch+1:03d}.pth")
            print(f"  ✅ Encoder weights saved: encoder_epoch_{epoch+1:03d}.pth")
            
            # 3. 最良モデルも保存
            if best_model_state is not None:
                torch.save(best_model_state, f"{checkpoint_dir}/best_model_epoch_{epoch+1:03d}.pth")
                print(f"  ✅ Best model saved: best_model_epoch_{epoch+1:03d}.pth")
    
    # 最終評価用のデータを取得（学習中の最後のバッチを使用）
    if last_batch is not None and last_mask_indices is not None:
        test_batch = last_batch.to(device)
        test_masked, test_mask_indices = mask_skeleton_joints(
            test_batch, mask_ratio=mask_ratio, mask_token=model.mask_token
        )
        print(f"Final evaluation using last training batch")
    else:
        print(f"Final evaluation using initial sample batch")
        test_batch = sample_batch.to(device)
        test_masked, test_mask_indices = mask_skeleton_joints(
            test_batch, mask_ratio=mask_ratio, mask_token=model.mask_token
        )
    
    print(f"Test batch shape: {test_batch.shape}")
    print(f"Test masked joints: {[len(mask_idx) for mask_idx in test_mask_indices]}")
    
    # 評価モードで推論
    model.eval()
    with torch.no_grad():
        # 座標空間での評価
        test_reconstructed = model(test_masked)
        final_loss, final_loss_dict = compute_reconstruction_loss(
            test_batch, test_reconstructed, test_mask_indices,
            loss_type=loss_fn, beta=beta
        )
    
    print(f"\nTraining completed!")
    print(f"Final test loss: {final_loss.item():.6f} (per joint: {final_loss.item()/22:.6f})")
    
    # 実際のマスク関節数で正規化
    final_masked_joints = np.mean([len(mask_idx) for mask_idx in test_mask_indices]) if test_mask_indices else 0
    final_unmasked_joints = 22 - final_masked_joints
    
    print(f"Final masked joints loss: {final_loss_dict['masked_joints_loss_mean']:.6f} (per joint: {final_loss_dict['masked_joints_loss_mean']/final_masked_joints:.6f})")
    print(f"Final unmasked joints loss: {final_loss_dict.get('unmasked_joints_loss_mean', 0.0):.6f} (per joint: {final_loss_dict.get('unmasked_joints_loss_mean', 0.0)/final_unmasked_joints:.6f})")
    print(f"Final masked joints: {final_masked_joints:.1f}, unmasked joints: {final_unmasked_joints:.1f}")
    
    # 最終的な重み保存
    if save_dir:
        print(f"\n💾 Saving final weights...")
        
        # 1. 最終モデルの全体重み
        final_checkpoint = {
            'epoch': max_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': final_loss.item(),
            'training_history': training_history,
            'config': {
                'mask_ratio': mask_ratio,
                'lr': lr,
                'weight_decay': weight_decay,
                'max_epochs': max_epochs
            }
        }
        torch.save(final_checkpoint, f"{checkpoint_dir}/final_model.pth")
        print(f"  ✅ Final model saved: final_model.pth")
        
        # 2. 最良モデルの最終保存
        if best_model_state is not None:
            torch.save(best_model_state, f"{checkpoint_dir}/best_model_final.pth")
            print(f"  ✅ Best model saved: best_model_final.pth")
        
        # 3. エンコーダーの最終重み
        encoder_state = {}
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_state[name] = param.data.clone()
        
        encoder_final = {
            'epoch': max_epochs,
            'encoder_state_dict': encoder_state,
            'final_loss': final_loss.item(),
            'config': {
                'mask_ratio': mask_ratio,
                'lr': lr,
                'weight_decay': weight_decay
            }
        }
        torch.save(encoder_final, f"{checkpoint_dir}/encoder_final.pth")
        print(f"  ✅ Final encoder saved: encoder_final.pth")
        
        print(f"\n📁 All weights saved in: {checkpoint_dir}/")
        print(f"   - checkpoint_epoch_020.pth, checkpoint_epoch_040.pth, ...")
        print(f"   - encoder_epoch_020.pth, encoder_epoch_040.pth, ...")
        print(f"   - best_model_epoch_020.pth, best_model_epoch_040.pth, ...")
        print(f"   - final_model.pth, best_model_final.pth, encoder_final.pth")
    
    # 学習結果の可視化（save_dirが指定されている場合）
    if save_dir:
        print(f"\nSaving training results to {save_dir}/")
        
        # 1. 学習曲線のプロット
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(training_history['epoch'], training_history['all_joints_loss'], 'b-', label='All Joints Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('All Joints Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(training_history['epoch'], training_history['masked_joints_loss_mean'], 'r-', label='Masked Joints Loss', linewidth=2)
        plt.fill_between(training_history['epoch'], 
                        np.array(training_history['masked_joints_loss_mean']) - np.array(training_history['masked_joints_loss_std']),
                        np.array(training_history['masked_joints_loss_mean']) + np.array(training_history['masked_joints_loss_std']),
                        alpha=0.3, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Masked Joints Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(training_history['epoch'], training_history['all_joints_loss'], 'b-', label='All Joints', linewidth=2)
        plt.plot(training_history['epoch'], training_history['masked_joints_loss_mean'], 'r-', label='Masked Joints', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 再構成結果の可視化
        print("Creating reconstruction visualization...")

        plot_reconstruction_comparison(
            test_batch.cpu(),
            test_masked.cpu(),
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/reconstruction_comparison.png",
            overlay=False
        )
        plot_reconstruction_comparison(
            test_batch.cpu(),
            test_masked.cpu(),
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/reconstruction_overlay.png",
            overlay=True
        )

        plot_sequence_reconstruction_comparison(
            test_batch.cpu(),
            test_masked.cpu(),
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/sequence_reconstruction_comparison.png",
            overlay=False
        )
        plot_sequence_reconstruction_comparison(
            test_batch.cpu(),
            test_masked.cpu(),
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/sequence_reconstruction_overlay.png",
            overlay=True
        )
    
    return model, training_history


def plot_reconstruction_comparison(original_data, masked_data, reconstructed_data, mask_indices, save_path, overlay=False):
    """
    再構成結果の比較可視化（元データ、マスクデータ、再構成データ）
    
    Args:
        original_data: 元のスケルトンデータ [batch_size, seq_len, num_joints, 3]
        masked_data: マスクされたスケルトンデータ [batch_size, seq_len, num_joints, 3]
        reconstructed_data: 再構成されたスケルトンデータ [batch_size, seq_len, num_joints, 3]
        mask_indices: マスクされた関節のインデックス
        save_path: 保存パス
        overlay: Trueの場合、同じ座標に重ねて表示
    """
    # 最初のバッチの全フレームを取得
    orig_sequence = to_numpy_array(original_data[0])  # [seq_len, num_joints, 3]
    mask_sequence = to_numpy_array(masked_data[0])
    recon_sequence = to_numpy_array(reconstructed_data[0])
    masked_ids = to_numpy_indices(mask_indices[0])
    masked_ids = np.sort(masked_ids)
    
    # 最初のフレームのみを取得（既存の表示用）
    orig_frame = orig_sequence[0]  # [num_joints, 3]
    mask_frame = mask_sequence[0]
    recon_frame = recon_sequence[0]
    
    # 座標変換（カメラ座標系: X下+, Y左+, Z奥+）
    orig_x, orig_y, orig_z = to_camera_coords(orig_frame)
    mask_x_raw, mask_y_raw, mask_z_raw = to_camera_coords(mask_frame)
    recon_x, recon_y, recon_z = to_camera_coords(recon_frame)

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

    if overlay:
        # 重ねて表示
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#fbfbfb')
        ax.grid(False)
        
        # 元のスケルトン（薄い青）
        legend_entries = []

        handle = ax.scatter(
            orig_x,
            orig_y,
            orig_z,
            color=PLOT_COLORS['original_cloud'],
            s=22,
            alpha=0.3,
            label='Original (reference)',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Original (reference)')

        # 観測済みジョイント
        if unmasked_indices.size > 0:
            handle = ax.scatter(
                mask_x[unmasked_indices],
                mask_y[unmasked_indices],
                mask_z[unmasked_indices],
                color=PLOT_COLORS['visible_joint'],
                s=55,
                alpha=0.9,
                label='Observed joints',
                depthshade=False,
            )
            _add_legend_entry(legend_entries, handle, 'Observed joints')

        if masked_ids.size > 0:
            # マスクトークン位置（参考）
            handle = ax.scatter(
                mask_x_raw[masked_ids],
                mask_y_raw[masked_ids],
                mask_z_raw[masked_ids],
                color=PLOT_COLORS['masked_placeholder'],
                s=40,
                marker='x',
                linewidths=1.1,
                alpha=0.75,
                label='Mask token',
                depthshade=False,
            )
            _add_legend_entry(legend_entries, handle, 'Mask token')

            # 元の位置
            handle = ax.scatter(
                orig_x[masked_ids],
                orig_y[masked_ids],
                orig_z[masked_ids],
                color=PLOT_COLORS['masked_joint'],
                s=80,
                marker='o',
                edgecolor='k',
                linewidths=0.6,
                alpha=0.9,
                label='Masked (original)',
                depthshade=False,
            )
            _add_legend_entry(legend_entries, handle, 'Masked (original)')

            # 再構成位置
            handle = ax.scatter(
                recon_x[masked_ids],
                recon_y[masked_ids],
                recon_z[masked_ids],
                color=PLOT_COLORS['masked_joint'],
                s=90,
                marker='^',
                edgecolor='k',
                linewidths=0.6,
                alpha=0.9,
                label='Masked (reconstructed)',
                depthshade=False,
            )
            _add_legend_entry(legend_entries, handle, 'Masked (reconstructed)')

            for idx in masked_ids:
                ax.plot(
                    [orig_x[idx], recon_x[idx]],
                    [orig_y[idx], recon_y[idx]],
                    [orig_z[idx], recon_z[idx]],
                    color='#7f7f7f',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.75,
                )

        # 再構成されたジョイント（観測可能なもの）
        if unmasked_indices.size > 0:
            handle = ax.scatter(
                recon_x[unmasked_indices],
                recon_y[unmasked_indices],
                recon_z[unmasked_indices],
                color=PLOT_COLORS['reconstructed_joint'],
                s=55,
                alpha=0.8,
                label='Reconstructed (visible)',
                depthshade=False,
            )
            _add_legend_entry(legend_entries, handle, 'Reconstructed (visible)')

        # 骨格
        for edge in SKELETON_EDGES:
            if edge[0] < orig_frame.shape[0] and edge[1] < orig_frame.shape[0]:
                ax.plot(
                    [orig_x[edge[0]], orig_x[edge[1]]],
                    [orig_y[edge[0]], orig_y[edge[1]]],
                    [orig_z[edge[0]], orig_z[edge[1]]],
                    color=PLOT_COLORS['original_edge'],
                    linewidth=1.1,
                    alpha=0.35,
                )
            if edge[0] < mask_frame.shape[0] and edge[1] < mask_frame.shape[0]:
                if edge[0] in unmasked_set and edge[1] in unmasked_set:
                    ax.plot(
                        [mask_x[edge[0]], mask_x[edge[1]]],
                        [mask_y[edge[0]], mask_y[edge[1]]],
                        [mask_z[edge[0]], mask_z[edge[1]]],
                        color=PLOT_COLORS['visible_edge'],
                        linewidth=2.0,
                        alpha=0.8,
                    )
            if edge[0] < recon_frame.shape[0] and edge[1] < recon_frame.shape[0]:
                ax.plot(
                    [recon_x[edge[0]], recon_x[edge[1]]],
                    [recon_y[edge[0]], recon_y[edge[1]]],
                    [recon_z[edge[0]], recon_z[edge[1]]],
                    color=PLOT_COLORS['reconstructed_edge'],
                    linewidth=1.8,
                    alpha=0.75,
                )

        ax.set_xlabel('X (down +)')
        ax.set_ylabel('Y (left +)')
        ax.set_zlabel('Z (forward +)')
        ax.set_title('Skeleton Reconstruction Comparison (Overlay)')
        set_axes_equal(
            ax,
            [orig_x, mask_x, recon_x, mask_x_raw[masked_ids]],
            [orig_y, mask_y, recon_y, mask_y_raw[masked_ids]],
            [orig_z, mask_z, recon_z, mask_z_raw[masked_ids]],
            margin=0.08,
        )
        set_camera_view(ax)
        handles, labels = _unique_legend_entries(legend_entries)
        ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0.0, 0.0, 0.85, 1.0])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # 分離表示
    fig = plt.figure(figsize=(16, 5))
    legend_entries = []

    # 1. 元のスケルトン
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('#fbfbfb')
    ax1.grid(False)
    handle = ax1.scatter(
        orig_x,
        orig_y,
        orig_z,
        color=PLOT_COLORS['original_cloud'],
        s=22,
        alpha=0.3,
        label='Original (reference)',
        depthshade=False,
    )
    _add_legend_entry(legend_entries, handle, 'Original (reference)')
    if masked_ids.size > 0:
        handle = ax1.scatter(
            orig_x[masked_ids],
            orig_y[masked_ids],
            orig_z[masked_ids],
            color=PLOT_COLORS['masked_joint'],
            s=85,
            marker='o',
            edgecolor='k',
            linewidths=0.6,
            alpha=0.95,
            label='Masked joints',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Masked joints')
    for edge in SKELETON_EDGES:
        if edge[0] < orig_frame.shape[0] and edge[1] < orig_frame.shape[0]:
            ax1.plot(
                [orig_x[edge[0]], orig_x[edge[1]]],
                [orig_y[edge[0]], orig_y[edge[1]]],
                [orig_z[edge[0]], orig_z[edge[1]]],
                color=PLOT_COLORS['original_edge'],
                linewidth=1.3,
                alpha=0.5,
            )

    ax1.set_xlabel('X (down +)')
    ax1.set_ylabel('Y (left +)')
    ax1.set_zlabel('Z (forward +)')
    ax1.set_title('Original Skeleton')
    set_axes_equal(
        ax1,
        [orig_x],
        [orig_y],
        [orig_z],
        margin=0.08,
    )
    set_camera_view(ax1)

    # 2. マスクされたスケルトン
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor('#fbfbfb')
    ax2.grid(False)

    if unmasked_indices.size > 0:
        handle = ax2.scatter(
            mask_x[unmasked_indices],
            mask_y[unmasked_indices],
            mask_z[unmasked_indices],
            color=PLOT_COLORS['visible_joint'],
            s=55,
            alpha=0.9,
            label='Observed joints',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Observed joints')
    if masked_ids.size > 0:
        handle = ax2.scatter(
            mask_x_raw[masked_ids],
            mask_y_raw[masked_ids],
            mask_z_raw[masked_ids],
            color=PLOT_COLORS['masked_placeholder'],
            s=40,
            marker='x',
            linewidths=1.2,
            alpha=0.75,
            label='Mask token',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Mask token')
        handle = ax2.scatter(
            orig_x[masked_ids],
            orig_y[masked_ids],
            orig_z[masked_ids],
            color=PLOT_COLORS['masked_joint'],
            s=85,
            marker='o',
            edgecolor='k',
            linewidths=0.6,
            alpha=0.95,
            label='Masked (original)',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Masked (original)')
    for edge in SKELETON_EDGES:
        if edge[0] in unmasked_set and edge[1] in unmasked_set:
            ax2.plot(
                [mask_x[edge[0]], mask_x[edge[1]]],
                [mask_y[edge[0]], mask_y[edge[1]]],
                [mask_z[edge[0]], mask_z[edge[1]]],
                color=PLOT_COLORS['visible_edge'],
                linewidth=2.0,
                alpha=0.8,
            )

    ax2.set_xlabel('X (down +)')
    ax2.set_ylabel('Y (left +)')
    ax2.set_zlabel('Z (forward +)')
    ax2.set_title(f'Masked Skeleton (masked joints: {masked_ids.tolist()})')
    set_axes_equal(
        ax2,
        [mask_x, mask_x_raw[masked_ids]],
        [mask_y, mask_y_raw[masked_ids]],
        [mask_z, mask_z_raw[masked_ids]],
        margin=0.08,
    )
    set_camera_view(ax2)

    # 3. 再構成されたスケルトン
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#fbfbfb')
    ax3.grid(False)
    if unmasked_indices.size > 0:
        handle = ax3.scatter(
            recon_x[unmasked_indices],
            recon_y[unmasked_indices],
            recon_z[unmasked_indices],
            color=PLOT_COLORS['reconstructed_joint'],
            s=55,
            alpha=0.85,
            label='Reconstructed (visible)',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Reconstructed (visible)')
    if masked_ids.size > 0:
        handle = ax3.scatter(
            recon_x[masked_ids],
            recon_y[masked_ids],
            recon_z[masked_ids],
            color=PLOT_COLORS['masked_joint'],
            s=90,
            marker='^',
            edgecolor='k',
            linewidths=0.6,
            alpha=0.95,
            label='Reconstructed (masked)',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Reconstructed (masked)')
        handle = ax3.scatter(
            orig_x[masked_ids],
            orig_y[masked_ids],
            orig_z[masked_ids],
            facecolor='none',
            edgecolor=PLOT_COLORS['masked_joint'],
            s=95,
            marker='o',
            linewidths=1.2,
            alpha=0.9,
            label='Masked (original)',
            depthshade=False,
        )
        _add_legend_entry(legend_entries, handle, 'Masked (original)')
        for idx in masked_ids:
            ax3.plot(
                [orig_x[idx], recon_x[idx]],
                [orig_y[idx], recon_y[idx]],
                [orig_z[idx], recon_z[idx]],
                color='#7f7f7f',
                linestyle='--',
                linewidth=1.0,
                alpha=0.75,
            )
    for edge in SKELETON_EDGES:
        if edge[0] < recon_frame.shape[0] and edge[1] < recon_frame.shape[0]:
            ax3.plot(
                [recon_x[edge[0]], recon_x[edge[1]]],
                [recon_y[edge[0]], recon_y[edge[1]]],
                [recon_z[edge[0]], recon_z[edge[1]]],
                color=PLOT_COLORS['reconstructed_edge'],
                linewidth=1.8,
                alpha=0.75,
            )

    ax3.set_xlabel('X (down +)')
    ax3.set_ylabel('Y (left +)')
    ax3.set_zlabel('Z (forward +)')
    ax3.set_title('Reconstructed Skeleton')
    set_axes_equal(
        ax3,
        [recon_x, orig_x[masked_ids]],
        [recon_y, orig_y[masked_ids]],
        [recon_z, orig_z[masked_ids]],
        margin=0.08,
    )
    set_camera_view(ax3)
    handles, labels = _unique_legend_entries(legend_entries)
    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            ncol=min(3, len(labels)),
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reconstruction comparison saved to {save_path}")


def plot_sequence_reconstruction_comparison(original_data, masked_data, reconstructed_data, mask_indices, save_path, overlay=False):
    """
    9フレームシーケンス全体の再構成結果比較可視化
    
    Args:
        original_data: 元のスケルトンデータ [batch_size, seq_len, num_joints, 3]
        masked_data: マスクされたスケルトンデータ [batch_size, seq_len, num_joints, 3]
        reconstructed_data: 再構成されたスケルトンデータ [batch_size, seq_len, num_joints, 3]
        mask_indices: マスクされた関節のインデックス
        save_path: 保存パス
        overlay: Trueの場合、同じ座標に重ねて表示
    """
    # 最初のバッチの全フレームを取得
    orig_sequence = to_numpy_array(original_data[0])  # [seq_len, num_joints, 3]
    mask_sequence = to_numpy_array(masked_data[0])
    recon_sequence = to_numpy_array(reconstructed_data[0])
    masked_ids = to_numpy_indices(mask_indices[0])
    masked_ids = np.sort(masked_ids)
    
    seq_len = orig_sequence.shape[0]
    num_joints = orig_sequence.shape[1]

    joint_idx = np.arange(num_joints)
    unmasked_mask = ~np.isin(joint_idx, masked_ids)
    unmasked_indices = joint_idx[unmasked_mask]
    unmasked_set = set(unmasked_indices.tolist())
    
    if overlay:
        # 重ねて表示: 3x3グリッドで9フレームを表示
        fig = plt.figure(figsize=(14, 14))
        legend_entries = []
        
        for frame_idx in range(seq_len):
            ax = fig.add_subplot(3, 3, frame_idx + 1, projection='3d')
            ax.set_facecolor('#fbfbfb')
            ax.grid(False)
            
            # 現在のフレームのデータ
            orig_frame = orig_sequence[frame_idx]
            mask_frame = mask_sequence[frame_idx]
            recon_frame = recon_sequence[frame_idx]
            
            # 座標変換（カメラ座標系）
            orig_x, orig_y, orig_z = to_camera_coords(orig_frame)
            mask_x_raw, mask_y_raw, mask_z_raw = to_camera_coords(mask_frame)
            recon_x, recon_y, recon_z = to_camera_coords(recon_frame)
            mask_x = mask_x_raw.copy()
            mask_y = mask_y_raw.copy()
            mask_z = mask_z_raw.copy()
            if masked_ids.size > 0:
                mask_x[masked_ids] = orig_x[masked_ids]
                mask_y[masked_ids] = orig_y[masked_ids]
                mask_z[masked_ids] = orig_z[masked_ids]
            
            # 元のスケルトン（薄い青）
            label = 'Original (reference)' if frame_idx == 0 else None
            handle = ax.scatter(
                orig_x,
                orig_y,
                orig_z,
                color=PLOT_COLORS['original_cloud'],
                s=18,
                alpha=0.28,
                label=label,
                depthshade=False,
            )
            if frame_idx == 0:
                _add_legend_entry(legend_entries, handle, label)
            
            # 観測済みジョイント
            if unmasked_indices.size > 0:
                label = 'Observed joints' if frame_idx == 0 else None
                handle = ax.scatter(
                    mask_x[unmasked_indices],
                    mask_y[unmasked_indices],
                    mask_z[unmasked_indices],
                    color=PLOT_COLORS['visible_joint'],
                    s=50,
                    alpha=0.9,
                    label=label,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, label)
            
            # マスクされた関節（赤いX）
            if masked_ids.size > 0:
                label = 'Mask token' if frame_idx == 0 else None
                handle = ax.scatter(
                    mask_x_raw[masked_ids],
                    mask_y_raw[masked_ids],
                    mask_z_raw[masked_ids],
                    color=PLOT_COLORS['masked_placeholder'],
                    s=36,
                    marker='x',
                    linewidths=1.0,
                    alpha=0.7,
                    label=label,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, label)
                label = 'Masked (original)' if frame_idx == 0 else None
                handle = ax.scatter(
                    orig_x[masked_ids],
                    orig_y[masked_ids],
                    orig_z[masked_ids],
                    color=PLOT_COLORS['masked_joint'],
                    s=65,
                    marker='o',
                    edgecolor='k',
                    linewidths=0.5,
                    alpha=0.95,
                    label=label,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, label)
                label = 'Masked (reconstructed)' if frame_idx == 0 else None
                handle = ax.scatter(
                    recon_x[masked_ids],
                    recon_y[masked_ids],
                    recon_z[masked_ids],
                    color=PLOT_COLORS['masked_joint'],
                    s=70,
                    marker='^',
                    edgecolor='k',
                    linewidths=0.5,
                    alpha=0.95,
                    label=label,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, label)
                for idx in masked_ids:
                    ax.plot(
                        [orig_x[idx], recon_x[idx]],
                        [orig_y[idx], recon_y[idx]],
                        [orig_z[idx], recon_z[idx]],
                        color='#7f7f7f',
                        linestyle='--',
                        linewidth=0.9,
                        alpha=0.7,
                    )
            
            # 再構成された関節（緑）
            if unmasked_indices.size > 0:
                label = 'Reconstructed (visible)' if frame_idx == 0 else None
                handle = ax.scatter(
                    recon_x[unmasked_indices],
                    recon_y[unmasked_indices],
                    recon_z[unmasked_indices],
                    color=PLOT_COLORS['reconstructed_joint'],
                    s=50,
                    alpha=0.85,
                    label=label,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, label)
            
            # 骨格をプロット（再構成データ）
            for edge in SKELETON_EDGES:
                if edge[0] < num_joints and edge[1] < num_joints:
                    ax.plot(
                        [orig_x[edge[0]], orig_x[edge[1]]],
                        [orig_y[edge[0]], orig_y[edge[1]]],
                        [orig_z[edge[0]], orig_z[edge[1]]],
                        color=PLOT_COLORS['original_edge'],
                        linewidth=0.9,
                        alpha=0.35,
                    )
                    if edge[0] in unmasked_set and edge[1] in unmasked_set:
                        ax.plot(
                            [mask_x[edge[0]], mask_x[edge[1]]],
                            [mask_y[edge[0]], mask_y[edge[1]]],
                            [mask_z[edge[0]], mask_z[edge[1]]],
                            color=PLOT_COLORS['visible_edge'],
                            linewidth=1.4,
                            alpha=0.75,
                        )
                    ax.plot(
                        [recon_x[edge[0]], recon_x[edge[1]]],
                        [recon_y[edge[0]], recon_y[edge[1]]],
                        [recon_z[edge[0]], recon_z[edge[1]]],
                        color=PLOT_COLORS['reconstructed_edge'],
                        linewidth=1.5,
                        alpha=0.75,
                    )
            
            # 軸設定
            ax.set_xlabel('X (down +)')
            ax.set_ylabel('Y (left +)')
            ax.set_zlabel('Z (forward +)')
            ax.set_title(f'Frame {frame_idx + 1}')
            
            set_axes_equal(
                ax,
                [orig_x, mask_x, recon_x, mask_x_raw[masked_ids]],
                [orig_y, mask_y, recon_y, mask_y_raw[masked_ids]],
                [orig_z, mask_z, recon_z, mask_z_raw[masked_ids]],
                margin=0.08,
            )
            set_camera_view(ax)
            
            if frame_idx == 0:
                handles, labels = _unique_legend_entries(legend_entries)
                ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=False)
    
    else:
        # 分離表示: 3x3グリッドで9フレームを表示
        fig = plt.figure(figsize=(20, 15))
        legend_entries = []
        
        for frame_idx in range(seq_len):
            # 元データ
            ax1 = fig.add_subplot(3, 9, frame_idx * 3 + 1, projection='3d')
            orig_frame = orig_sequence[frame_idx]
            orig_x, orig_y, orig_z = to_camera_coords(orig_frame)
            ax1.set_facecolor('#fbfbfb')
            ax1.grid(False)
            
            if frame_idx == 0:
                label = 'Original (reference)'
            else:
                label = None
            handle = ax1.scatter(
                orig_x,
                orig_y,
                orig_z,
                color=PLOT_COLORS['original_cloud'],
                s=22,
                alpha=0.3,
                label=label,
                depthshade=False,
            )
            if frame_idx == 0:
                _add_legend_entry(legend_entries, handle, label)
            if masked_ids.size > 0:
                handle = ax1.scatter(
                    orig_x[masked_ids],
                    orig_y[masked_ids],
                    orig_z[masked_ids],
                    color=PLOT_COLORS['masked_joint'],
                    s=75,
                    marker='o',
                    edgecolor='k',
                    linewidths=0.5,
                    alpha=0.95,
                    label='Masked joints' if frame_idx == 0 else None,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, 'Masked joints')
            for edge in SKELETON_EDGES:
                if edge[0] < num_joints and edge[1] < num_joints:
                    ax1.plot(
                        [orig_x[edge[0]], orig_x[edge[1]]],
                        [orig_y[edge[0]], orig_y[edge[1]]],
                        [orig_z[edge[0]], orig_z[edge[1]]],
                        color=PLOT_COLORS['original_edge'],
                        linewidth=1.2,
                        alpha=0.45,
                    )
            
            ax1.set_title(f'Original F{frame_idx+1}')
            ax1.set_xlabel('X (down +)')
            ax1.set_ylabel('Y (left +)')
            ax1.set_zlabel('Z (forward +)')
            set_axes_equal(
                ax1,
                [orig_x],
                [orig_y],
                [orig_z],
                margin=0.08,
            )
            set_camera_view(ax1)
            
            # マスクデータ
            ax2 = fig.add_subplot(3, 9, frame_idx * 3 + 2, projection='3d')
            mask_frame = mask_sequence[frame_idx]
            mask_x_raw, mask_y_raw, mask_z_raw = to_camera_coords(mask_frame)
            ax2.set_facecolor('#fbfbfb')
            ax2.grid(False)
            mask_x = mask_x_raw.copy()
            mask_y = mask_y_raw.copy()
            mask_z = mask_z_raw.copy()
            if masked_ids.size > 0:
                mask_x[masked_ids] = orig_x[masked_ids]
                mask_y[masked_ids] = orig_y[masked_ids]
                mask_z[masked_ids] = orig_z[masked_ids]
            
            if unmasked_indices.size > 0:
                handle = ax2.scatter(
                    mask_x[unmasked_indices],
                    mask_y[unmasked_indices],
                    mask_z[unmasked_indices],
                    color=PLOT_COLORS['visible_joint'],
                    s=40,
                    alpha=0.9,
                    label='Observed' if frame_idx == 0 else None,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, 'Observed')
            if masked_ids.size > 0:
                handle = ax2.scatter(
                    mask_x_raw[masked_ids],
                    mask_y_raw[masked_ids],
                    mask_z_raw[masked_ids],
                    color=PLOT_COLORS['masked_placeholder'],
                    s=38,
                    marker='x',
                    linewidths=1.0,
                    alpha=0.7,
                    label='Mask token' if frame_idx == 0 else None,
                    depthshade=False,
                )
                if frame_idx == 0:
                    _add_legend_entry(legend_entries, handle, 'Mask token')
                ax2.scatter(
                    orig_x[masked_ids],
                    orig_y[masked_ids],
                    orig_z[masked_ids],
                    color=PLOT_COLORS['masked_joint'],
                    s=80,
                    marker='o',
                    edgecolor='k',
                    linewidths=0.5,
                    alpha=0.95,
                    label='Masked (orig)' if frame_idx == 0 else None,
                    depthshade=False,
                )
            for edge in SKELETON_EDGES:
                if edge[0] in unmasked_set and edge[1] in unmasked_set:
                    ax2.plot(
                        [mask_x[edge[0]], mask_x[edge[1]]],
                        [mask_y[edge[0]], mask_y[edge[1]]],
                        [mask_z[edge[0]], mask_z[edge[1]]],
                        color=PLOT_COLORS['visible_edge'],
                        linewidth=1.6,
                        alpha=0.8,
                    )
            
            ax2.set_title(f'Masked F{frame_idx+1}')
            ax2.set_xlabel('X (down +)')
            ax2.set_ylabel('Y (left +)')
            ax2.set_zlabel('Z (forward +)')
            set_axes_equal(
                ax2,
                [mask_x, mask_x_raw[masked_ids]],
                [mask_y, mask_y_raw[masked_ids]],
                [mask_z, mask_z_raw[masked_ids]],
                margin=0.08,
            )
            set_camera_view(ax2)
            
            # 再構成データ
            ax3 = fig.add_subplot(3, 9, frame_idx * 3 + 3, projection='3d')
            recon_frame = recon_sequence[frame_idx]
            recon_x, recon_y, recon_z = to_camera_coords(recon_frame)
            ax3.set_facecolor('#fbfbfb')
            ax3.grid(False)
            if unmasked_indices.size > 0:
                ax3.scatter(
                    recon_x[unmasked_indices],
                    recon_y[unmasked_indices],
                    recon_z[unmasked_indices],
                    color=PLOT_COLORS['reconstructed_joint'],
                    s=35,
                    alpha=0.9,
                    label='Reconstructed' if frame_idx == 0 else None,
                    depthshade=False,
                )
            if masked_ids.size > 0:
                ax3.scatter(
                    recon_x[masked_ids],
                    recon_y[masked_ids],
                    recon_z[masked_ids],
                    color=PLOT_COLORS['masked_joint'],
                    s=80,
                    marker='^',
                    edgecolor='k',
                    linewidths=0.5,
                    alpha=0.95,
                    label='Recon masked' if frame_idx == 0 else None,
                    depthshade=False,
                )
                ax3.scatter(
                    orig_x[masked_ids],
                    orig_y[masked_ids],
                    orig_z[masked_ids],
                    facecolor='none',
                    edgecolor=PLOT_COLORS['masked_joint'],
                    s=90,
                    marker='o',
                    linewidths=1.0,
                    alpha=0.9,
                    label='Masked (orig)' if frame_idx == 0 else None,
                    depthshade=False,
                )
            for edge in SKELETON_EDGES:
                if edge[0] < num_joints and edge[1] < num_joints:
                    ax3.plot(
                        [recon_x[edge[0]], recon_x[edge[1]]],
                        [recon_y[edge[0]], recon_y[edge[1]]],
                        [recon_z[edge[0]], recon_z[edge[1]]],
                        color=PLOT_COLORS['reconstructed_edge'],
                        linewidth=1.6,
                        alpha=0.75,
                    )
            
            ax3.set_title(f'Reconstructed F{frame_idx+1}')
            ax3.set_xlabel('X (down +)')
            ax3.set_ylabel('Y (left +)')
            ax3.set_zlabel('Z (forward +)')
            set_axes_equal(
                ax3,
                [recon_x, orig_x[masked_ids]],
                [recon_y, orig_y[masked_ids]],
                [recon_z, orig_z[masked_ids]],
                margin=0.08,
            )
            set_camera_view(ax3)

        handles, labels = _unique_legend_entries(legend_entries)
        if handles:
            fig.legend(
                handles,
                labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.99),
                frameon=False,
                ncol=min(3, len(labels)),
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sequence reconstruction comparison saved to {save_path}")


class SkeletonDataset(Dataset):
    """JTA-3DPデータセット用のPyTorch Dataset"""
    
    def __init__(self, json_files, track_size=16, sequence_length=9, num_joints=22, frequency=1):
        self.json_files = json_files
        self.track_size = track_size
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.frequency = frequency
        self.data = []
        
        print(f"Loading {len(json_files)} JSON files...")
        self.file_stats = []  # ファイルごとの統計情報
        for json_file in tqdm(json_files):
            patches_before = len(self.data)
            self.load_jta_data_from_json(json_file)
            patches_after = len(self.data)
            patches_added = patches_after - patches_before
            
            self.file_stats.append({
                'file': os.path.basename(json_file),
                'patches_added': patches_added,
                'total_patches': patches_after
            })
        
        print(f"Total patches loaded: {len(self.data)}")
    
    def load_jta_data_from_json(self, json_file):
        """JTA-3DP JSONファイルからスケルトンデータを読み込み"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # JTA-3DPの形式: [[frame_data1], [frame_data2], ...]
        # 各frame_data: [frame_id, person_id, bird_eye_x, bird_eye_y, x0,...,x21, y0,...,y21, z0,...,z21]
        if not isinstance(data, list) or len(data) == 0:
            print(f"Invalid data format in {json_file}")
            return
        
        # 各フレームのデータを解析
        frames_data = []
        for frame_data in data:
            if not isinstance(frame_data, list) or len(frame_data) < 4 + self.num_joints * 3:
                print(f"Invalid frame data format in {json_file}")
                continue
                
            frame_id = frame_data[0]
            person_id = frame_data[1]
            bird_eye_x = frame_data[2]  # 鳥瞰図視点のX座標
            bird_eye_y = frame_data[3]  # 鳥瞰図視点のY座標
            
            # 関節座標を抽出: [x0,...,x21, y0,...,y21, z0,...,z21]
            coords_start = 4  # frame_id, person_id, bird_eye_x, bird_eye_yをスキップ
            x_coords = frame_data[coords_start:coords_start + self.num_joints]  # x0,...,x21
            y_coords = frame_data[coords_start + self.num_joints:coords_start + self.num_joints * 2]  # y0,...,y21
            z_coords = frame_data[coords_start + self.num_joints * 2:coords_start + self.num_joints * 3]  # z0,...,z21
            
            # 3D座標を結合
            joints_3d = np.array([x_coords, y_coords, z_coords]).T  # [num_joints, 3]
            
            # 有効なデータかチェック（全て0でない、NaNでない）
            if not np.allclose(joints_3d, 0) and not np.isnan(joints_3d).any():
                frames_data.append({
                    'frame_id': frame_id,
                    'person_id': person_id,
                    'bird_eye_x': bird_eye_x,
                    'bird_eye_y': bird_eye_y,
                    'joints': joints_3d
                })
        
        # トラックごとにデータを整理
        if len(frames_data) > 0:
            self.create_tracks_from_frames(frames_data)
        else:
            print(f"No valid frames found in {json_file}")
    
    def create_tracks_from_frames(self, frames_data):
        """フレームデータからトラックを作成"""
        # 人物IDごとにグループ化
        person_dict = {}
        for frame_data in frames_data:
            person_id = frame_data['person_id']
            if person_id not in person_dict:
                person_dict[person_id] = []
            person_dict[person_id].append(frame_data)
        
        # 各人物からシーケンスを作成
        for person_id, person_frames in person_dict.items():
            if len(person_frames) >= self.track_size * self.frequency:
                # フレームを時系列でソート
                person_frames.sort(key=lambda x: x['frame_id'])
                
                # トラックサイズ分のフレームを抽出
                for start_idx in range(0, len(person_frames) - self.track_size * self.frequency + 1, self.track_size):
                    end_idx = start_idx + self.track_size * self.frequency
                    
                    # フレームをサンプリング
                    sampled_frames = person_frames[start_idx:end_idx:self.frequency]
                    
                    if len(sampled_frames) >= self.sequence_length:
                        # 関節座標を抽出
                        joints_sequence = []
                        for frame in sampled_frames[:self.sequence_length]:  # シーケンス長に制限
                            joints_sequence.append(frame['joints'])
                        
                        if len(joints_sequence) == self.sequence_length:
                            patch = np.array(joints_sequence)  # [sequence_length, num_joints, 3]
                            self.data.append(patch)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patch = self.data[idx]  # [sequence_length, num_joints, 3]
        return torch.FloatTensor(patch)

def load_json_files(data_dir):
    """指定ディレクトリからJSONファイルを検索"""
    json_files = []
    
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
    else:
        print(f"Data directory {data_dir} not found. Using dummy data.")
    
    return json_files


def create_visualization_directory():
    """可視化用のディレクトリを作成"""
    os.makedirs("skeleton/train", exist_ok=True)
    print("Created visualization directory: skeleton/train/")

def main(args):
    print("=" * 60)
    print("SKELETON DATA LOADING AND PATCHING DEMO")
    print("=" * 60)
    
    # YAML設定を読み込み
    config_path = "configs_skeleton.yml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        print(f"Warning: {config_path} not found, using default values")
        config = {}
    
    # 可視化ディレクトリを作成
    create_visualization_directory()
    
    # パラメータ設定（YAML設定を優先、なければデフォルト値）
    patch_size = config.get('TRAIN', {}).get('input_track_size', 16)  # パッチサイズ
    sequence_length = 9  # シーケンス長
    num_joints = 22      # ジョイント数
    batch_size = config.get('MODEL', {}).get('batch_size', 4)  # バッチサイズ
    data_dir = config.get('DATA', {}).get('input_dir', "data/jta_3dp_row") + "/train/"  # JSONデータのディレクトリ
    track_size = config.get('TRAIN', {}).get('track_size', 9)  # トラックサイズ
    frequency = config.get('TRAIN', {}).get('frequency', 1)  # フレームサンプリング頻度
    save_dir = "skeleton/train"  # 可視化保存ディレクトリ
    
    # 学習パラメータ (coordinate pretraining)
    lr = config.get('MODEL', {}).get('lr', 0.001)
    max_epochs = config.get('MODEL', {}).get('max_epoch', 100)
    mask_ratio = config.get('MODEL', {}).get('mask_rate', 0.05)
    weight_decay = config.get('MODEL', {}).get('weight_decay', 0.01)
    optimizer_name = config.get('TRAIN', {}).get('optimizer', 'adam')
    loss_fn = config.get('MODEL', {}).get('loss_fn', 'mse')
    beta = config.get('MODEL', {}).get('beta', 2.0)
    feature_dim = config.get('MODEL', {}).get('feature_dim', 64)

    
    print(f"Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Mask ratio: {mask_ratio}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Optimizer: {optimizer_name}")
    print(f"  - Loss function: {loss_fn}")
    if loss_fn == 'rce':
        print(f"  - Beta (RCE): {beta}")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Track size: {track_size}")
    print(f"  - Frequency: {frequency}")
    print(f"  - Data directory: {data_dir}")
    
    # JSONファイルを検索
    print(f"Searching for JSON files in {data_dir}...")
    json_files = load_json_files(data_dir)
    
    if len(json_files) == 0:
        print("No JSON files found. Generating dummy data for demonstration...")
        
    else:
        print(f"Found {len(json_files)} JSON files. Creating dataset...")
        
        dataset = SkeletonDataset(
            json_files=json_files,
            track_size=track_size,
            sequence_length=sequence_length,
            num_joints=num_joints,
            frequency=frequency
        )
        
        print(f"Dataset created with {len(dataset)} patches")
            
    # DataLoaderを作成（YAML設定を使用）
    num_workers = config.get('TRAIN', {}).get('num_workers', 0)
    print(f"\nCreating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    sample_batch = next(iter(dataloader))
    print(f"Sample batch shape: {sample_batch.shape}")
    
    # マスキング機能のテスト
    print("\nTesting joint masking...")
    test_batch = next(iter(dataloader))
    masked_data, mask_indices = mask_skeleton_joints(test_batch, mask_ratio=mask_ratio)
    print(f"Masked {len(mask_indices[0])} joints per sample")
    
    # Pretraining実行
    print("\nStarting skeleton pretraining...")
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Pretraining実行（YAML設定を使用）
    trained_model, training_history = skeleton_pretrain(
        dataloader=dataloader,
        device=device,
        mask_ratio=mask_ratio,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=save_dir,
        loss_fn=loss_fn,
        beta=beta,
        feature_dim=feature_dim
    )
    
    print("\nPretraining completed!")

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs_skeleton.yml")
    print(args)
    main(args)
