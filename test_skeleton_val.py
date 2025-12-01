import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

from main_skeleton import (
    STGCN18Reconstructor,
    SkeletonDataset,
    mask_skeleton_joints,
    aggregate_batch_errors_from_distances,
)


def load_checkpoint_flex(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded full model_state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        enc_state = ckpt["encoder_state_dict"]
        # そのまま読み込めるキーのみ適用
        model_state = model.state_dict()
        filtered = {k: v for k, v in enc_state.items() if k in model_state}
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded encoder_state_dict into model. Copied params: {len(filtered)}")
    else:
        # 直接 state_dict と仮定
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded raw state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def build_val_dataloader(config, batch_size=4, num_workers=0):
    input_dir = config.get("DATA", {}).get("input_dir", "data/jta_3dp_row")
    data_dir = os.path.join(input_dir, "val")
    track_size = config.get("TRAIN", {}).get("track_size", 9)
    sequence_length = config.get("TRAIN", {}).get("input_track_size", 9)
    num_joints = 22
    frequency = config.get("TRAIN", {}).get("frequency", 1)

    # JSON 検索
    json_files = []
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.json'):
                    json_files.append(os.path.join(root, f))
    print(f"VAL json files: {len(json_files)} in {data_dir}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    dataset = SkeletonDataset(
        json_files=json_files,
        track_size=track_size,
        sequence_length=sequence_length,
        num_joints=num_joints,
        frequency=frequency,
    )
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl


def plot_skeleton_reconstruction(batch, reconstructed_coords, mask_indices, save_dir, batch_idx, num_joints=22):
    """
    各サンプルのスケルトンを可視化（9フレームを同じ3Dプロットに重ねて表示）
    - 元の関節（マスクされていない）: 緑
    - 元の関節（マスクされた）: 赤
    - 再構成された関節: オレンジ
    
    main_skeleton.pyのplot_reconstruction_comparisonを参考に実装
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # スケルトンの骨格接続定義（main_skeleton.pyから引用）
    skeleton_edges = [
        # 頭部
        (0, 1), (1, 2),  # head_top -> head_center -> neck
        
        # 右腕
        (2, 3), (3, 4), (4, 5), (5, 6),  # neck -> right_clavicle -> right_shoulder -> right_elbow -> right_wrist
        
        # 左腕
        (2, 7), (7, 8), (8, 9), (9, 10),  # neck -> left_clavicle -> left_shoulder -> left_elbow -> left_wrist
        
        # 脊椎
        (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # neck -> spine0 -> spine1 -> spine2 -> spine3 -> spine4
        
        # 右足
        (15, 16), (16, 17), (17, 18),  # spine4 -> right_hip -> right_knee -> right_ankle
        
        # 左足
        (15, 19), (19, 20), (20, 21),  # spine4 -> left_hip -> left_knee -> left_ankle
    ]
    
    batch_size = batch.shape[0]
    seq_len = batch.shape[1]
    
    for sample_idx in range(batch_size):
        # 1つの3Dプロットに全フレームを表示
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # マスク情報（テンソルをリストに変換してからsetに）
        if sample_idx < len(mask_indices):
            mask_tensor = mask_indices[sample_idx]
            if isinstance(mask_tensor, torch.Tensor):
                mask_set = set(mask_tensor.cpu().tolist())
            else:
                mask_set = set(mask_tensor)
        else:
            mask_set = set()
        
        # デバッグ: マスク情報を確認
        if sample_idx == 0 and batch_idx == 0:
            print(f"Debug: Sample {sample_idx}, Masked joints: {mask_set}, Total joints: {num_joints}")
        
        # 全フレームの座標を収集（座標範囲の計算用）
        all_orig_x, all_orig_y, all_orig_z = [], [], []
        all_recon_x, all_recon_y, all_recon_z = [], [], []
        
        # ラベル用フラグ（最初の1回だけラベルを表示）
        masked_labeled = False
        unmasked_labeled = False
        recon_labeled = False
        
        # 各フレームをプロット
        for frame_idx in range(seq_len):
            # 元の関節座標
            original_joints = batch[sample_idx, frame_idx].cpu().numpy()  # [V, 3]
            # 再構成された関節座標
            recon_joints = reconstructed_coords[sample_idx, frame_idx].cpu().numpy()  # [V, 3]
            
            # 座標変換: X=右方向+, Y=下方向+, Z=奥方向+（main_skeleton.pyと同様）
            orig_x = -original_joints[:, 1]
            orig_y = -original_joints[:, 0]
            orig_z = original_joints[:, 2]
            
            recon_x = -recon_joints[:, 1]
            recon_y = -recon_joints[:, 0]
            recon_z = recon_joints[:, 2]
            
            # 座標範囲計算用に保存
            all_orig_x.extend(orig_x)
            all_orig_y.extend(orig_y)
            all_orig_z.extend(orig_z)
            all_recon_x.extend(recon_x)
            all_recon_y.extend(recon_y)
            all_recon_z.extend(recon_z)
            
            # フレームごとに色を濃くする（0.3から1.0に徐々に）
            color_intensity = 0.3 + (frame_idx / max(seq_len - 1, 1)) * 0.7  # 0.3 ~ 1.0
            
            # 透明度もフレームごとに変える（最初は薄く、徐々に濃く）
            alpha_base = 0.4 + (frame_idx / max(seq_len - 1, 1)) * 0.5  # 0.4 ~ 0.9
            
            # 元の関節をプロット（マスク/非マスクで色分け）
            masked_count = 0
            for joint_idx in range(min(num_joints, original_joints.shape[0])):
                if joint_idx in mask_set:
                    masked_count += 1
                    # デバッグ: 最初のフレームでマスクされた関節を表示
                    if frame_idx == 0 and sample_idx == 0 and batch_idx == 0:
                        print(f"Debug: Plotting masked joint {joint_idx} at frame {frame_idx}")
                    # マスクされた関節: 鮮やかな赤で、目立つマーカー（常に濃い赤）
                    label = 'Masked (original)' if frame_idx == 0 and not masked_labeled else ''
                    # 常に鮮やかな赤（フレームごとの濃さに関係なく）
                    mask_color = (color_intensity, 0.0, 0.0, alpha_base)  # 鮮やかな赤
                    # マスクされた関節は大きく、Xマーカーで表示
                    ax.scatter(orig_x[joint_idx], orig_y[joint_idx], orig_z[joint_idx], 
                              c=[mask_color], s=50, marker='o', alpha=0.9, 
                              edgecolors='darkred', linewidths=2, label=label)
                    if frame_idx == 0 and not masked_labeled:
                        masked_labeled = True
                else:
                    # マスクされていない関節: 青で、フレームごとに濃く
                    label = 'Unmasked (original)' if frame_idx == 0 and not unmasked_labeled else ''
                    # 青を徐々に濃く: (R, G, B, A) = (0, 0, color_intensity, alpha)
                    blue_color = (0.0, 0.0, color_intensity, alpha_base)
                    ax.scatter(orig_x[joint_idx], orig_y[joint_idx], orig_z[joint_idx], 
                              c=[blue_color], s=50, marker='o', alpha=alpha_base, label=label)
                    if frame_idx == 0 and not unmasked_labeled:
                        unmasked_labeled = True
            
            # 再構成された関節をプロット（緑で、フレームごとに濃く）
            recon_masked_labeled = False
            recon_unmasked_labeled = False
            for joint_idx in range(min(num_joints, recon_joints.shape[0])):
                if joint_idx in mask_set:
                    # マスクされていた関節の再構成結果: より目立つように表示
                    label = 'Reconstructed (masked)' if frame_idx == 0 and not recon_masked_labeled else ''
                    # マスクされていた関節の再構成は、より鮮やかな緑で、大きく表示
                    recon_masked_color = (0.0, color_intensity, 0.0)  # 鮮やかな緑
                    ax.scatter(recon_x[joint_idx], recon_y[joint_idx], recon_z[joint_idx], 
                              c=[recon_masked_color], s=50, marker='^', alpha=0.9,
                              edgecolors='darkgreen', linewidths=2, label=label)
                    if frame_idx == 0 and not recon_masked_labeled:
                        recon_masked_labeled = True
                else:
                    # マスクされていなかった関節の再構成: 通常の緑で表示
                    label = 'Reconstructed (unmasked)' if frame_idx == 0 and not recon_unmasked_labeled else ''
                    # 緑を徐々に濃く: (R, G, B, A) = (0, color_intensity, 0, alpha)
                    green_color = (0.0, color_intensity, 0.0, alpha_base * 0.9)
                    ax.scatter(recon_x[joint_idx], recon_y[joint_idx], recon_z[joint_idx], 
                              c=[green_color], s=50, marker='o', alpha=alpha_base * 0.9, label=label)
                    if frame_idx == 0 and not recon_unmasked_labeled:
                        recon_unmasked_labeled = True
            
            # ラベル管理用に更新
            if frame_idx == 0:
                if recon_masked_labeled or recon_unmasked_labeled:
                    recon_labeled = True
            
            # スケルトン接続線をプロット（フレームごとに濃く）
            line_alpha = 0.4 + (frame_idx / max(seq_len - 1, 1)) * 0.5  # 0.4 ~ 0.9
            line_width = 1.5 + (frame_idx / max(seq_len - 1, 1)) * 1.0  # 1.5 ~ 2.5（最後のフレームが太い）
            
            for edge in skeleton_edges:
                if edge[0] < original_joints.shape[0] and edge[1] < original_joints.shape[0]:
                    # マスクされた関節に接続している線かチェック
                    is_masked_edge = (edge[0] in mask_set) or (edge[1] in mask_set)
                    
                    if is_masked_edge:
                        # マスクされた関節に接続している線: 赤で強調表示
                        mask_line_color = (color_intensity, 0.0, 0.0)  # 鮮やかな赤
                        ax.plot([orig_x[edge[0]], orig_x[edge[1]]],
                               [orig_y[edge[0]], orig_y[edge[1]]],
                               [orig_z[edge[0]], orig_z[edge[1]]],
                               color=mask_line_color, linewidth=line_width + 1.0, 
                               alpha=0.9, linestyle='--')  # 破線で強調
                    else:
                        # 元のスケルトンの骨格線（青で、フレームごとに濃く）
                        blue_line_color = (0.0, 0.0, color_intensity)
                        ax.plot([orig_x[edge[0]], orig_x[edge[1]]],
                               [orig_y[edge[0]], orig_y[edge[1]]],
                               [orig_z[edge[0]], orig_z[edge[1]]],
                               color=blue_line_color, linewidth=line_width, alpha=line_alpha)
                
                if edge[0] < recon_joints.shape[0] and edge[1] < recon_joints.shape[0]:
                    # 再構成されたスケルトンの骨格線（緑で、フレームごとに濃く）
                    green_line_color = (0.0, color_intensity, 0.0)
                    ax.plot([recon_x[edge[0]], recon_x[edge[1]]],
                           [recon_y[edge[0]], recon_y[edge[1]]],
                           [recon_z[edge[0]], recon_z[edge[1]]],
                           color=green_line_color, linewidth=line_width, alpha=line_alpha * 0.9)
            
            # デバッグ: 最初のフレームでマスクされた関節の数を表示
            if frame_idx == 0 and sample_idx == 0 and batch_idx == 0:
                print(f"Debug: Frame {frame_idx}, Found {masked_count} masked joints out of {min(num_joints, original_joints.shape[0])} joints")
        
        # 座標範囲を統一
        all_x = all_orig_x + all_recon_x
        all_y = all_orig_y + all_recon_y
        all_z = all_orig_z + all_recon_z
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)
        
        # マージンを追加
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        z_margin = (z_max - z_min) * 0.1
        
        ax.set_xlabel('X+')
        ax.set_ylabel('Y+')
        ax.set_zlabel('Z+')
        ax.set_title(f'Batch {batch_idx}, Sample {sample_idx} - All {seq_len} Frames Overlaid\n'
                    f'Blue: Original, Red: Masked joints, Green: Reconstructed (darker = later frames)', 
                    fontsize=12)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        
        # main_skeleton.pyと同様の視点
        ax.view_init(elev=90, azim=180)
        
        # 凡例を表示
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(save_dir, f'batch_{batch_idx:03d}_sample_{sample_idx:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {save_path}")


def create_feature_to_coord_inverse(model):
    """
    coord_to_featureの重みの転置を使って、feature_dim → in_channelsの逆変換層を作成
    """
    coord_to_feat = model.coord_to_feature
    # 転置重みを使った逆変換層を作成
    inverse_layer = nn.Linear(coord_to_feat.out_features, coord_to_feat.in_features, bias=False)
    # coord_to_featureの重みの転置を設定
    with torch.no_grad():
        inverse_layer.weight.data = coord_to_feat.weight.data.t()  # [in_channels, gcn_input_dim] -> [gcn_input_dim, in_channels]
    return inverse_layer


def evaluate_coordinate_reconstruction(model, dataloader, device, mask_ratio=0.15, max_batches=5, save_dir=None):
    """
    特徴空間での再構成を検証:
    1. batch -> coord_to_feature -> original_features
    2. batch + mask_indices -> model(return_features=True) -> reconstructed_features
    3. reconstructed_features -> feature_to_coord_inverse -> reconstructed_coords
    
    Args:
        save_dir: 可視化画像の保存ディレクトリ（Noneの場合は保存しない）
    """
    model.eval()
    
    # coord_to_featureの逆変換層を作成（重みの転置を使用）
    feature_to_coord_inverse = create_feature_to_coord_inverse(model).to(device)
    
    all_distances = []  # list of [B,T,V]
    all_mask_indices = []  # list per batch of list per sample
    
    with torch.no_grad():
        for bidx, batch in enumerate(dataloader):
            if bidx >= max_batches:
                break
            batch = batch.to(device)  # [B,T,V,3]
            
            # 1. マスキング
            masked_batch, mask_indices = mask_skeleton_joints(batch, mask_ratio=mask_ratio)
            
            # 3. マスクされたデータで特徴再構成
            reconstructed_features = model(batch, mask_indices=mask_indices, return_features=True)  # [B,T,V,gcn_input_dim]
            
            # 4. 特徴→座標変換（coord_to_featureの逆変換、重みの転置を使用）
            reconstructed_coords = feature_to_coord_inverse(reconstructed_features)  # [B,T,V,3]
            
            # 可視化（全バッチ、全サンプル）
            if save_dir is not None:
                plot_skeleton_reconstruction(batch, reconstructed_coords, mask_indices, save_dir, bidx)
            
            # 5. 距離計算（各関節の再構成誤差を計算）
            # batch: 元の座標 [B,T,V,3] (B=batch, T=time, V=joints, 3=x,y,z)
            # reconstructed_coords: 再構成された座標 [B,T,V,3]
            # batch - reconstructed_coords: 各関節の座標差分 [B,T,V,3]
            # torch.norm(..., dim=-1): 最後の次元(3次元座標)に対してユークリッド距離を計算
            # → 結果: 各関節の距離誤差 [B,T,V] (各関節ごとに1つの距離値)
            distances = torch.norm(batch - reconstructed_coords, dim=-1)  # [B,T,V]
            
            # バッチごとの距離をリストに保存（CPUに移動してメモリ節約）
            all_distances.append(distances.cpu())
            
            # マスク情報も保存（後でマスクされた関節とマスクされていない関節を区別して評価するため）
            # mask_indices: 各サンプルごとにマスクされた関節のインデックスリスト
            all_mask_indices.extend(mask_indices)
    
    if len(all_distances) == 0:
        raise RuntimeError("No batches evaluated")
    
    # 全バッチの距離を結合: [B1,T,V] + [B2,T,V] + ... → [N,T,V] (N=全サンプル数)
    distances_cat = torch.cat(all_distances, dim=0)  # [N,T,V]
    overall_mean_error = float(distances_cat.mean().item())
    
    # マスクされた関節とマスクされていない関節を区別して統計を計算
    # - masked: マスクされた関節の再構成誤差（学習対象）
    # - unmasked: マスクされていない関節の再構成誤差（参考値）
    stats = aggregate_batch_errors_from_distances(distances_cat, all_mask_indices)
    stats["overall_mean_error"] = overall_mean_error
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--cfg", type=str, default="configs_skeleton.yml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="skeleton/val_visualizations", help="Directory to save visualization images")
    args = parser.parse_args()

    # 設定読み込み
    if os.path.exists(args.cfg):
        with open(args.cfg, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    device = torch.device(args.device)

    # モデル（座標再構成）
    feature_dim = config.get('MODEL', {}).get('feature_dim', 128)
    model = STGCN18Reconstructor(
        in_channels=3,
        out_channels=3,
        gcn_input_dim=64,   # 座標再構成（ST-GCN in_channels=3）
        feature_dim=feature_dim,
    ).to(device)

    # チェックポイント読み込み
    load_checkpoint_flex(model, args.ckpt, map_location=device)

    # データローダ
    dl = build_val_dataloader(config, batch_size=args.batch_size, num_workers=args.num_workers)

    # 評価（可視化も含む）
    print(f"\nSaving visualizations to: {args.save_dir}")
    stats = evaluate_coordinate_reconstruction(
        model, dl, device, 
        mask_ratio=args.mask_ratio, 
        max_batches=args.max_batches,
        save_dir=args.save_dir
    )

    print("\n=== Coordinate Reconstruction on VAL ===")
    print(f"Overall mean error: {stats['overall_mean_error']:.4f} m")
    print(f"Masked micro mean:  {stats['micro_masked_mean']:.4f} m")
    print(f"Unmasked micro mean:{stats['micro_unmasked_mean']:.4f} m")
    print(f"Masked macro mean:  {stats['macro_masked_mean']:.4f} m")
    print(f"Unmasked macro mean:{stats['macro_unmasked_mean']:.4f} m")
    print(f"Masked instances:   {stats['total_masked_instances']}")
    print(f"Unmasked instances: {stats['total_unmasked_instances']}")
    print(f"Avg masked/sample:  {stats['avg_masked_per_sample']:.2f} ({stats['mask_rate']*100:.1f}%)")


# ============================================================================
# 下流タスクでエンコーダー部分だけを使う例
# ============================================================================

def freeze_encoder(model, freeze_encoder=True, freeze_coord_to_feature=True):
    """
    エンコーダー部分を凍結する
    
    Args:
        model: STGCN18Reconstructorモデル
        freeze_encoder: Trueの場合、encoderを凍結
        freeze_coord_to_feature: Trueの場合、coord_to_featureを凍結
    
    Returns:
        frozen_params: 凍結されたパラメータの数
        trainable_params: 学習可能なパラメータの数
    """
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'encoder' in name and freeze_encoder:
            param.requires_grad = False
            frozen_params += param.numel()
        elif 'coord_to_feature' in name and freeze_coord_to_feature:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()
    
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return frozen_params, trainable_params


def create_downstream_model(base_model, num_classes, freeze_encoder_flag=True):
    """
    下流タスク用のモデルを作成（エンコーダーを凍結して新しい分類ヘッドを追加）
    
    Args:
        base_model: 事前学習済みのSTGCN18Reconstructorモデル
        num_classes: 分類クラス数
        freeze_encoder_flag: Trueの場合、エンコーダーを凍結
    
    Returns:
        downstream_model: 下流タスク用のモデル
    """
    import torch.nn as nn
    
    # エンコーダーを凍結
    if freeze_encoder_flag:
        freeze_encoder(base_model, freeze_encoder=True, freeze_coord_to_feature=True)
    
    # 下流タスク用の分類ヘッドを作成
    feature_dim = base_model.feature_dim
    device = next(base_model.parameters()).device
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # モデル全体をラップ
    class DownstreamModel(nn.Module):
        def __init__(self, encoder_model, classifier):
            super().__init__()
            self.encoder = encoder_model.encoder
            self.coord_to_feature = encoder_model.coord_to_feature
            self.classifier = classifier
        
        def forward(self, x):
            # 座標→特徴変換
            batch_size, seq_len, num_joints, coords = x.shape
            coord_features = self.coord_to_feature(x)  # [B, T, V, gcn_input_dim]
            
            # ST-GCNに渡すために形状を変換
            features_for_stgcn = coord_features.permute(0, 3, 1, 2)  # [B, gcn_input_dim, T, V]
            
            # エンコーダーで特徴抽出
            stgcn_features, _ = self.encoder.extract_feature(features_for_stgcn)  # [B, feature_dim, T, V]
            
            # グローバルプーリング（時空間の平均）
            pooled_features = stgcn_features.mean(dim=(2, 3))  # [B, feature_dim]
            
            # 分類
            logits = self.classifier(pooled_features)  # [B, num_classes]
            return logits
    
    return DownstreamModel(base_model, classifier)


# ============================================================================
# 使用例
# ============================================================================
"""
エンコーダー部分だけのチェックポイント（encoder_state_dict）を使ったfine-tuning例:

```python
import torch
import torch.nn as nn
from main_skeleton import STGCN18Reconstructor
from test_skeleton_val import load_checkpoint_flex, freeze_encoder, create_downstream_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 基本モデルを作成
feature_dim = 256
base_model = STGCN18Reconstructor(
    in_channels=3,
    out_channels=3,
    gcn_input_dim=64,
    feature_dim=feature_dim,
).to(device)

# 2. エンコーダー部分のみのチェックポイントを読み込む
load_checkpoint_flex(base_model, "path/to/encoder_epoch_XXX.pth", map_location=device)

# 3. エンコーダーを凍結
freeze_encoder(base_model, freeze_encoder=True, freeze_coord_to_feature=True)

# 4. 下流タスク用のモデルを作成（オプション: 分類ヘッド付き）
num_classes = 10
downstream_model = create_downstream_model(base_model, num_classes, freeze_encoder_flag=True)

# 5. 学習（エンコーダーは凍結、分類ヘッドのみ学習）
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, downstream_model.parameters()),
    lr=1e-3
)

# 学習ループ
for batch, labels in dataloader:
    optimizer.zero_grad()
    logits = downstream_model(batch)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
```

注意:
- encoder_state_dictには encoder と coord_to_feature のみが保存されている
- freeze_encoder()でエンコーダー部分を凍結できる
- 下流タスクでは通常、エンコーダーから特徴を抽出して新しいタスクヘッドを学習する
"""

if __name__ == "__main__":
    main()
