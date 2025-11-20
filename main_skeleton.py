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
from graphmae.models.st_gcn_aaai18 import ST_GCN_18

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


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
    masked_joints = mask_indices[0].cpu().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 座標変換（CPUに移動してからNumPy変換）
    orig_x = -original_patch[0, :, 1].cpu().numpy()
    orig_y = -original_patch[0, :, 0].cpu().numpy()
    orig_z = original_patch[0, :, 2].cpu().numpy()
    
    mask_x = -masked_patch[0, :, 1].cpu().numpy()
    mask_y = -masked_patch[0, :, 0].cpu().numpy()
    mask_z = masked_patch[0, :, 2].cpu().numpy()
    
    # マスクされていない関節とマスクされた関節を分離（NumPy統一）
    unmasked_mask = ~np.isin(np.arange(len(masked_patch[0])), masked_joints)
    masked_mask = np.isin(np.arange(len(masked_patch[0])), masked_joints)
    
    # 1. 元のスケルトン全体（薄い青）
    ax.scatter(orig_x, orig_y, orig_z, c='lightblue', s=30, alpha=0.4, label='Original skeleton')
    
    # 2. マスクされていない関節（青）
    ax.scatter(mask_x[unmasked_mask], mask_y[unmasked_mask], mask_z[unmasked_mask], 
               c='blue', s=60, alpha=0.8, label='Unmasked joints')
    
    # 3. マスクされた関節の元の位置（オレンジ）
    ax.scatter(orig_x[masked_joints], orig_y[masked_joints], orig_z[masked_joints], 
               c='orange', s=80, alpha=0.9, label='Masked joints (original position)')
    
    # 4. マスクされた関節の現在位置（赤いX）
    ax.scatter(mask_x[masked_mask], mask_y[masked_mask], mask_z[masked_mask], 
               c='red', s=100, marker='x', alpha=0.9, label='Masked joints (masked position)')
    
    # 骨格接続の定義
    skeleton_edges = [
        (0, 1), (1, 2),  # 頭部
        (2, 3), (3, 4), (4, 5), (5, 6),  # 右腕
        (2, 7), (7, 8), (8, 9), (9, 10),  # 左腕
        (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # 脊椎
        (15, 16), (16, 17), (17, 18),  # 右足
        (15, 19), (19, 20), (20, 21),  # 左足
    ]
    
    # 骨格接続をプロット
    for edge in skeleton_edges:
        if edge[0] < len(original_patch[0]) and edge[1] < len(original_patch[0]):
            # 元の骨格（薄い色）
            x1_orig, x2_orig = -original_patch[0, edge[0], 1], -original_patch[0, edge[1], 1]
            y1_orig, y2_orig = -original_patch[0, edge[0], 0], -original_patch[0, edge[1], 0]
            z1_orig, z2_orig = original_patch[0, edge[0], 2], original_patch[0, edge[1], 2]
            ax.plot([x1_orig, x2_orig], [y1_orig, y2_orig], [z1_orig, z2_orig], 
                   'lightgray', linewidth=1, alpha=0.3)
            
            # マスクされていない骨格（青）
            if edge[0] not in masked_joints and edge[1] not in masked_joints:
                x1_mask, x2_mask = -masked_patch[0, edge[0], 1], -masked_patch[0, edge[1], 1]
                y1_mask, y2_mask = -masked_patch[0, edge[0], 0], -masked_patch[0, edge[1], 0]
                z1_mask, z2_mask = masked_patch[0, edge[0], 2], masked_patch[0, edge[1], 2]
                ax.plot([x1_mask, x2_mask], [y1_mask, y2_mask], [z1_mask, z2_mask], 
                       'blue', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X+')
    ax.set_ylabel('Y+')
    ax.set_zlabel('Z+')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 軸の範囲を統一
    all_x = np.concatenate([orig_x, mask_x])
    all_y = np.concatenate([orig_y, mask_y])
    all_z = np.concatenate([orig_z, mask_z])
    
    margin = 0.5
    ax.set_xlim([all_x.min()-margin, all_x.max()+margin])
    ax.set_ylim([all_y.min()-margin, all_y.max()+margin])
    ax.set_zlim([all_z.min()-margin, all_z.max()+margin])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")
    print(f"Masked joints: {masked_joints}")




def compute_feature_reconstruction_loss(original_features, reconstructed_features, mask_indices, loss_type='rce', beta=2.0):
    """
    特徴空間での再構成損失を計算
    
    Args:
        original_features: 元の特徴 [batch_size, seq_len, num_joints, feature_dim]
        reconstructed_features: 再構成された特徴 [batch_size, seq_len, num_joints, feature_dim]
        mask_indices: マスクされた関節のインデックス
        loss_type: 損失関数のタイプ ('mse', 'l1', 'rce')
        beta: RCEの重み付けパラメータ（β ≥ 1）
    
    Returns:
        total_loss: 総損失
        loss_dict: 損失の詳細情報
    """
    device = original_features.device
    batch_size = original_features.shape[0]
    
    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction='none')
    elif loss_type == 'rce':
        # RCE (Re-weighted Cosine Error) の実装
        def rce_loss(x, y, beta=beta):
            """
            Re-weighted Cosine Error for features
            LRCE = Σ(1/|V| - (xT·y)/(|V|×||x||×||y||))^β
            """
            batch_size, seq_len, num_joints, feature_dim = x.shape
            
            x_flat = x.view(batch_size, seq_len, num_joints, feature_dim)
            y_flat = y.view(batch_size, seq_len, num_joints, feature_dim)
            x_norm = torch.norm(x_flat, dim=3, keepdim=True)
            y_norm = torch.norm(y_flat, dim=3, keepdim=True)
            cosine_sim = torch.sum(x_flat * y_flat, dim=3, keepdim=True) / (x_norm * y_norm + 1e-8)
            rce = torch.pow(1 - cosine_sim, beta)
            return rce.expand(batch_size, seq_len, num_joints, feature_dim)
        
        loss_fn = rce_loss
    else:
        loss_fn = nn.MSELoss(reduction='none')
    
    all_losses = loss_fn(reconstructed_features, original_features)
    
    masked_losses = []
    unmasked_losses = []
    
    for b in range(batch_size):
        masked_joints = mask_indices[b]
        if isinstance(masked_joints, torch.Tensor):
            masked_joints = masked_joints.to(device)
        else:
            masked_joints = torch.tensor(masked_joints, device=device)
        
        all_joints = torch.arange(original_features.shape[2], device=device)
        unmasked_joints = all_joints[~torch.isin(all_joints, masked_joints)]
        
        if len(masked_joints) > 0:
            masked_losses.append(all_losses[b, :, masked_joints, :].mean())
        if len(unmasked_joints) > 0:
            unmasked_losses.append(all_losses[b, :, unmasked_joints, :].mean())
    
    masked_avg_loss = torch.stack(masked_losses).mean() if masked_losses else torch.tensor(0.0, device=device)
    unmasked_avg_loss = torch.stack(unmasked_losses).mean() if unmasked_losses else torch.tensor(0.0, device=device)
    
    total_loss = masked_avg_loss + unmasked_avg_loss
    
    num_masked_joints = np.mean([len(mask_idx) for mask_idx in mask_indices]) if mask_indices else 0
    num_unmasked_joints = original_features.shape[2] - num_masked_joints
    
    total_loss_per_joint = total_loss.item() / original_features.shape[2]
    masked_loss_per_joint = masked_avg_loss.item() / num_masked_joints if num_masked_joints > 0 else 0.0
    unmasked_loss_per_joint = unmasked_avg_loss.item() / num_unmasked_joints if num_unmasked_joints > 0 else 0.0
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'total_loss_per_joint': total_loss_per_joint,
        'masked_joints_loss_mean': masked_avg_loss.item(),
        'masked_joints_loss_std': torch.stack(masked_losses).std().item() if masked_losses else 0.0,
        'masked_joints_loss_per_joint': masked_loss_per_joint,
        'unmasked_joints_loss_mean': unmasked_avg_loss.item(),
        'unmasked_joints_loss_std': torch.stack(unmasked_losses).std().item() if unmasked_losses else 0.0,
        'unmasked_joints_loss_per_joint': unmasked_loss_per_joint,
        'num_masked_joints': num_masked_joints,
        'num_unmasked_joints': num_unmasked_joints
    }
    
    return total_loss, loss_dict



class STGCN18Reconstructor(nn.Module):
    """
    ST-GCN-18をエンコーダーとして使用するスケルトン再構成器（特徴空間での再構成）
    
    新しいアプローチ:
    1. 座標→特徴変換
    2. 特徴空間でマスキング（学習可能なマスクトークン使用）
    3. ST-GCN-18でグラフ特徴抽出
    4. 特徴空間で再構成
    5. 学習可能なLinear層で特徴→座標変換
    """
    def __init__(self, in_channels=3, out_channels=3, gcn_input_dim=64, feature_dim=256):
        super(STGCN18Reconstructor, self).__init__()
        
        self.feature_dim = feature_dim
        self.gcn_input_dim = gcn_input_dim
        
        # 学習可能なマスクトークン
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, gcn_input_dim))
        
        # ST-GCN-18をエンコーダーとして使用
        # 22関節用のグラフ設定
        graph_cfg = {
            'layout': 'jta_3dp_row',
            'strategy': 'distance',
            'max_hop': 1,
            'dilation': 1
        }
        
        # ST-GCN-18エンコーダー（座標特徴の次元を入力として使用）
        self.encoder = ST_GCN_18(
            in_channels= gcn_input_dim,  # 座標特徴の次元を使用
            feature_dim=feature_dim,
            graph_cfg=graph_cfg,
            edge_importance_weighting=True,
            data_bn=True
        )   
        
        # 座標→特徴変換層（論文通り: linearly transformed）
        self.coord_to_feature = nn.Linear(in_channels, gcn_input_dim)
        
        # 特徴→座標変換層（学習可能なLinear層）
        self.feature_to_coord = nn.Linear(gcn_input_dim, out_channels)
        
        # 特徴空間での再構成デコーダー
        self.feature_decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # ST-GCN特徴のみ
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, gcn_input_dim),
            nn.ReLU()
        )
        

    
    def forward(self, x, mask_indices=None, return_features=False, return_original_features=False):
        """
        Args:
            x: [batch_size, seq_len, num_joints, 3] 座標データ
            mask_indices: マスクされた関節のインデックス（特徴空間マスキング用）
            return_features: Trueの場合、特徴も返す
            return_original_features: Trueの場合、元の特徴も返す
        Returns:
            reconstructed: [batch_size, seq_len, num_joints, 3] (可視化用)
            features: [batch_size, seq_len, num_joints, feature_dim] (学習用、return_features=True時のみ)
            original_features: [batch_size, seq_len, num_joints, feature_dim] (元の特徴、return_original_features=True時のみ)
        """
        batch_size, seq_len, num_joints, coords = x.shape
        
        # 1. 座標→特徴変換
        coord_features = self.coord_to_feature(x)  # [batch_size, seq_len, num_joints, feature_dim]
        
        # 2. 特徴空間でマスキング
        masked_features, _ = mask_skeleton_joints(coord_features, mask_token=self.mask_token, mask_indices=mask_indices)
                
        # 4. ST-GCN-18でグラフ特徴抽出
        stgcn_features, _ = self.encoder.extract_feature(masked_features.permute(0, 3, 1, 2))
        stgcn_features = stgcn_features.permute(0, 2, 3, 1)
        reconstructed_features_flat = self.feature_decoder(stgcn_features.reshape(-1, self.feature_dim))
        reconstructed_features = reconstructed_features_flat.view(batch_size, seq_len, num_joints, self.gcn_input_dim)
        
        return reconstructed_features



def calculate_joint_errors(*args, **kwargs):
    raise RuntimeError("Coordinate reconstruction metrics have been removed from main_skeleton.py")


def calculate_masked_unmasked_batch_errors(*args, **kwargs):
    raise RuntimeError("Coordinate reconstruction metrics have been removed from main_skeleton.py")


def aggregate_batch_errors_from_distances(*args, **kwargs):
    raise RuntimeError("Coordinate reconstruction metrics have been removed from main_skeleton.py")


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
    model = STGCN18Reconstructor(in_channels=3, out_channels=3, gcn_input_dim=64, feature_dim=feature_dim).to(device)
    print(f"Model initialized:")
    print(f"  - Model type: ST-GCN-18 Feature Reconstructor")
    print(f"  - Encoder: ST-GCN-18")
    print(f"  - Input channels: 3")
    print(f"  - Output channels: 3")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Reconstruction: Feature space")
    
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
    fixed_masked, fixed_mask_indices = mask_skeleton_joints(fixed_batch, mask_ratio=mask_ratio)
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
            masked_batch, mask_indices = mask_skeleton_joints(batch, mask_ratio=mask_ratio)
            
            # 学習: 特徴空間での再構成（座標→特徴→マスキング→ST-GCN→特徴再構成）
            # 1. 元の特徴を取得（非マスク）
            original_features = model.coord_to_feature(batch)  # [batch_size, seq_len, num_joints, feature_dim]
                
            # 2. マスクされた特徴で再構成
            reconstructed_features = model(batch, mask_indices=mask_indices, return_features=True)
            # 損失計算（特徴空間での損失、学習対象）
            loss, loss_dict = compute_feature_reconstruction_loss(
                    original_features,  # 元の特徴（非マスク）
                    reconstructed_features,  # 再構成された特徴
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
                fixed_reconstructed = model(fixed_batch, mask_indices=fixed_mask_indices, return_features=False)

            fixed_masked_for_viz, _ = mask_skeleton_joints(fixed_batch, mask_token=0.0, mask_indices=fixed_mask_indices)
            reconstruction_history.append({
                'epoch': epoch + 1,
                'original': fixed_batch.cpu(),
                'masked': fixed_masked_for_viz.cpu(),
                'reconstructed': fixed_reconstructed.cpu(),
                'mask_indices': fixed_mask_indices
            })

            if save_dir:
                plot_path = f"{save_dir}/reconstruction_epoch_{epoch+1:03d}.png"
                plot_reconstruction_comparison(
                    fixed_batch,
                    fixed_masked_for_viz,
                    fixed_reconstructed,
                    fixed_mask_indices,
                    plot_path
                )
                print(f"  ✅ Reconstruction plot saved: reconstruction_epoch_{epoch+1:03d}.png")

                sequence_plot_path = f"{save_dir}/sequence_reconstruction_epoch_{epoch+1:03d}.png"
                plot_sequence_reconstruction_comparison(
                    fixed_batch,
                    fixed_masked_for_viz,
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
        test_masked, test_mask_indices = mask_skeleton_joints(test_batch, mask_ratio=mask_ratio)
        print(f"Final evaluation using last training batch")
    else:
        print(f"Final evaluation using initial sample batch")
        test_batch = sample_batch.to(device)
        test_masked, test_mask_indices = mask_skeleton_joints(test_batch, mask_ratio=mask_ratio)
    
    print(f"Test batch shape: {test_batch.shape}")
    print(f"Test masked joints: {[len(mask_idx) for mask_idx in test_mask_indices]}")
    
    # 評価モードで推論
    model.eval()
    with torch.no_grad():
        # 特徴空間での評価
        test_original_features = model.coord_to_feature(test_batch)
        test_reconstructed_features = model(test_batch, mask_indices=test_mask_indices, return_features=True)
        final_loss, final_loss_dict = compute_feature_reconstruction_loss(
            test_original_features, test_reconstructed_features, test_mask_indices,
            loss_type=loss_fn, beta=beta
        )
        # 可視化用に座標を取得
        test_reconstructed = model(test_batch, mask_indices=test_mask_indices, return_features=False)
    
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
        
        # 3. エンコーダーとcoord_to_featureの最終重み
        encoder_state = {}
        for name, param in model.named_parameters():
            if 'encoder' in name or 'coord_to_feature' in name:
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
        # 特徴空間マスキングの場合、マスクされた座標を生成して可視化
        test_masked_for_viz, _ = mask_skeleton_joints(test_batch, mask_token=0.0, mask_indices=test_mask_indices)
        
        # 1フレーム表示（既存）
        plot_reconstruction_comparison(
            test_batch.cpu(),
            test_masked_for_viz.cpu(), 
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/reconstruction_comparison.png",
            overlay=False
        )
        # 1フレーム重ねて表示（既存）
        plot_reconstruction_comparison(
            test_batch.cpu(),
            test_masked_for_viz.cpu(), 
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/reconstruction_overlay.png",
            overlay=True
        )
        
        # 9フレームシーケンス表示（新規）
        plot_sequence_reconstruction_comparison(
            test_batch.cpu(),
            test_masked_for_viz.cpu(), 
            test_reconstructed.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/sequence_reconstruction_comparison.png",
            overlay=False
        )

        # 9フレームシーケンス重ねて表示（新規）
        plot_sequence_reconstruction_comparison(
            test_batch.cpu(),
            test_masked_for_viz.cpu(), 
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
    orig_sequence = original_data[0].detach().cpu().numpy()  # [seq_len, num_joints, 3]
    mask_sequence = masked_data[0].detach().cpu().numpy()
    recon_sequence = reconstructed_data[0].detach().cpu().numpy()
    masked_joints = mask_indices[0].detach().cpu().numpy()
    
    # 最初のフレームのみを取得（既存の表示用）
    orig_frame = orig_sequence[0]  # [num_joints, 3]
    mask_frame = mask_sequence[0]
    recon_frame = recon_sequence[0]
    
    # 座標変換: X=右方向+, Y=下方向+, Z=奥方向+
    orig_x = -orig_frame[:, 1]
    orig_y = -orig_frame[:, 0]
    orig_z = orig_frame[:, 2]
    
    mask_x = -mask_frame[:, 1]
    mask_y = -mask_frame[:, 0]
    mask_z = mask_frame[:, 2]
    
    recon_x = -recon_frame[:, 1]
    recon_y = -recon_frame[:, 0]
    recon_z = recon_frame[:, 2]
    
    # スケルトンの骨格接続定義
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
    
    # 座標範囲を統一
    all_x = np.concatenate([orig_x, mask_x, recon_x])
    all_y = np.concatenate([orig_y, mask_y, recon_y])
    all_z = np.concatenate([orig_z, mask_z, recon_z])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    # マージンを追加
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    z_margin = (z_max - z_min) * 0.1
    
    if overlay:
        # 重ねて表示
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 元のスケルトン（薄い青）
        ax.scatter(orig_x, orig_y, orig_z, c='lightblue', s=30, alpha=0.6, label='Original')
        
        # 骨格をプロット（元データ）
        for edge in skeleton_edges:
            if edge[0] < len(orig_frame) and edge[1] < len(orig_frame):
                x1, x2 = -orig_frame[edge[0], 1], -orig_frame[edge[1], 1]
                y1, y2 = -orig_frame[edge[0], 0], -orig_frame[edge[1], 0]
                z1, z2 = orig_frame[edge[0], 2], orig_frame[edge[1], 2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'lightblue', linewidth=1, alpha=0.5)
        
        # マスクされた関節（赤いX）
        masked_mask = np.isin(np.arange(len(mask_frame)), masked_joints)
        ax.scatter(mask_x[masked_mask], mask_y[masked_mask], mask_z[masked_mask], 
                   c='red', s=100, marker='x', label='Masked joints', linewidth=3)
        
        # 再構成された関節（緑）
        ax.scatter(recon_x, recon_y, recon_z, c='green', s=50, alpha=0.8, label='Reconstructed')
        
        # マスクされた関節の再構成結果を強調（オレンジ）
        ax.scatter(recon_x[masked_joints], recon_y[masked_joints], recon_z[masked_joints], 
                   c='orange', s=100, marker='o', label='Reconstructed masked joints', alpha=0.9)
        
        # 骨格をプロット（再構成データ）
        for edge in skeleton_edges:
            if edge[0] < len(recon_frame) and edge[1] < len(recon_frame):
                x1, x2 = -recon_frame[edge[0], 1], -recon_frame[edge[1], 1]
                y1, y2 = -recon_frame[edge[0], 0], -recon_frame[edge[1], 0]
                z1, z2 = recon_frame[edge[0], 2], recon_frame[edge[1], 2]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'green', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('X+')
        ax.set_ylabel('Y+')
        ax.set_zlabel('Z+')
        ax.set_title('Skeleton Reconstruction Comparison (Overlay)')
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.view_init(elev=90, azim=180)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # 分離表示
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 元のスケルトン
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(orig_x, orig_y, orig_z, c='blue', s=50, alpha=0.7, label='Original joints')
    
    # 骨格をプロット
    for edge in skeleton_edges:
        if edge[0] < len(orig_frame) and edge[1] < len(orig_frame):
            x1, x2 = -orig_frame[edge[0], 1], -orig_frame[edge[1], 1]
            y1, y2 = -orig_frame[edge[0], 0], -orig_frame[edge[1], 0]
            z1, z2 = orig_frame[edge[0], 2], orig_frame[edge[1], 2]
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('X+')
    ax1.set_ylabel('Y+')
    ax1.set_zlabel('Z+')
    ax1.set_title('Original Skeleton')
    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax1.set_zlim(z_min - z_margin, z_max + z_margin)
    ax1.view_init(elev=90, azim=180)
    
    # 2. マスクされたスケルトン
    ax2 = fig.add_subplot(132, projection='3d')
    
    # マスクされていない関節（NumPy統一）
    unmasked_mask = ~np.isin(np.arange(len(mask_frame)), masked_joints)
    ax2.scatter(mask_x[unmasked_mask], mask_y[unmasked_mask], mask_z[unmasked_mask], 
               c='blue', s=50, alpha=0.7, label='Unmasked joints')
    
    # マスクされた関節（0に設定された関節）
    masked_mask = np.isin(np.arange(len(mask_frame)), masked_joints)
    ax2.scatter(mask_x[masked_mask], mask_y[masked_mask], mask_z[masked_mask], 
               c='red', s=100, marker='x', label='Masked joints')
    
    # 骨格をプロット（マスクされた関節は別の色でプロット）
    for edge in skeleton_edges:
        if (edge[0] < len(mask_frame) and edge[1] < len(mask_frame) and
            edge[0] not in masked_joints and edge[1] not in masked_joints):
            x1, x2 = -mask_frame[edge[0], 1], -mask_frame[edge[1], 1]
            y1, y2 = -mask_frame[edge[0], 0], -mask_frame[edge[1], 0]
            z1, z2 = mask_frame[edge[0], 2], mask_frame[edge[1], 2]
            ax2.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('X+')
    ax2.set_ylabel('Y+')
    ax2.set_zlabel('Z+')
    ax2.set_title(f'Masked Skeleton (Masked joints: {masked_joints})')
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_zlim(z_min - z_margin, z_max + z_margin)
    
    ax2.view_init(elev=90, azim=180)
    ax2.legend()
    
    # 3. 再構成されたスケルトン
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 全ての関節を表示
    ax3.scatter(recon_x, recon_y, recon_z, c='green', s=50, alpha=0.7, label='Reconstructed joints')
    
    # マスクされた関節を強調
    ax3.scatter(recon_x[masked_joints], recon_y[masked_joints], recon_z[masked_joints], 
               c='orange', s=100, marker='o', label='Reconstructed masked joints')
    
    # 骨格をプロット
    for edge in skeleton_edges:
        if edge[0] < len(recon_frame) and edge[1] < len(recon_frame):
            x1, x2 = -recon_frame[edge[0], 1], -recon_frame[edge[1], 1]
            y1, y2 = -recon_frame[edge[0], 0], -recon_frame[edge[1], 0]
            z1, z2 = recon_frame[edge[0], 2], recon_frame[edge[1], 2]
            ax3.plot([x1, x2], [y1, y2], [z1, z2], 'g-', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('X+')
    ax3.set_ylabel('Y+')
    ax3.set_zlabel('Z+')
    ax3.set_title('Reconstructed Skeleton')
    ax3.set_xlim(x_min - x_margin, x_max + x_margin)
    ax3.set_ylim(y_min - y_margin, y_max + y_margin)
    ax3.set_zlim(z_min - z_margin, z_max + z_margin)
    ax3.view_init(elev=90, azim=180)
    ax3.legend()
    
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
    orig_sequence = original_data[0].cpu().numpy()  # [seq_len, num_joints, 3]
    mask_sequence = masked_data[0].cpu().numpy()
    recon_sequence = reconstructed_data[0].cpu().numpy()
    masked_joints = mask_indices[0].cpu().numpy()
    
    seq_len = orig_sequence.shape[0]
    
    # スケルトンの骨格接続定義
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
    
    if overlay:
        # 重ねて表示: 3x3グリッドで9フレームを表示
        fig = plt.figure(figsize=(15, 15))
        
        for frame_idx in range(seq_len):
            ax = fig.add_subplot(3, 3, frame_idx + 1, projection='3d')
            
            # 現在のフレームのデータ
            orig_frame = orig_sequence[frame_idx]
            mask_frame = mask_sequence[frame_idx]
            recon_frame = recon_sequence[frame_idx]
            
            # 座標変換
            orig_x = -orig_frame[:, 1]
            orig_y = -orig_frame[:, 0]
            orig_z = orig_frame[:, 2]
            
            mask_x = -mask_frame[:, 1]
            mask_y = -mask_frame[:, 0]
            mask_z = mask_frame[:, 2]
            
            recon_x = -recon_frame[:, 1]
            recon_y = -recon_frame[:, 0]
            recon_z = recon_frame[:, 2]
            
            # 元のスケルトン（薄い青）
            ax.scatter(orig_x, orig_y, orig_z, c='lightblue', s=20, alpha=0.6, label='Original')
            
            # 骨格をプロット（元データ）
            for edge in skeleton_edges:
                if edge[0] < len(orig_frame) and edge[1] < len(orig_frame):
                    x1, x2 = -orig_frame[edge[0], 1], -orig_frame[edge[1], 1]
                    y1, y2 = -orig_frame[edge[0], 0], -orig_frame[edge[1], 0]
                    z1, z2 = orig_frame[edge[0], 2], orig_frame[edge[1], 2]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'lightblue', linewidth=1, alpha=0.4)
            
            # マスクされた関節（赤いX）
            masked_mask = np.isin(np.arange(len(mask_frame)), masked_joints)
            ax.scatter(mask_x[masked_mask], mask_y[masked_mask], mask_z[masked_mask], 
                       c='red', s=60, marker='x', label='Masked', linewidth=2)
            
            # 再構成された関節（緑）
            ax.scatter(recon_x, recon_y, recon_z, c='green', s=30, alpha=0.8, label='Reconstructed')
            
            # マスクされた関節の再構成結果を強調（オレンジ）
            ax.scatter(recon_x[masked_joints], recon_y[masked_joints], recon_z[masked_joints], 
                       c='orange', s=60, marker='o', label='Recon masked', alpha=0.9)
            
            # 骨格をプロット（再構成データ）
            for edge in skeleton_edges:
                if edge[0] < len(recon_frame) and edge[1] < len(recon_frame):
                    x1, x2 = -recon_frame[edge[0], 1], -recon_frame[edge[1], 1]
                    y1, y2 = -recon_frame[edge[0], 0], -recon_frame[edge[1], 0]
                    z1, z2 = recon_frame[edge[0], 2], recon_frame[edge[1], 2]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'green', linewidth=2, alpha=0.7)
            
            # 軸設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx + 1}')
            
            # 軸の範囲を統一
            all_x = np.concatenate([orig_x, mask_x, recon_x])
            all_y = np.concatenate([orig_y, mask_y, recon_y])
            all_z = np.concatenate([orig_z, mask_z, recon_z])
            
            margin = 0.1
            ax.set_xlim([all_x.min()-margin, all_x.max()+margin])
            ax.set_ylim([all_y.min()-margin, all_y.max()+margin])
            ax.set_zlim([all_z.min()-margin, all_z.max()+margin])
            
            # 最初のフレームのみ凡例を表示
            if frame_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    else:
        # 分離表示: 3x3グリッドで9フレームを表示
        fig = plt.figure(figsize=(20, 15))
        
        for frame_idx in range(seq_len):
            # 元データ
            ax1 = fig.add_subplot(3, 9, frame_idx * 3 + 1, projection='3d')
            orig_frame = orig_sequence[frame_idx]
            orig_x = -orig_frame[:, 1]
            orig_y = -orig_frame[:, 0]
            orig_z = orig_frame[:, 2]
            
            ax1.scatter(orig_x, orig_y, orig_z, c='blue', s=30, alpha=0.8)
            for edge in skeleton_edges:
                if edge[0] < len(orig_frame) and edge[1] < len(orig_frame):
                    x1, x2 = -orig_frame[edge[0], 1], -orig_frame[edge[1], 1]
                    y1, y2 = -orig_frame[edge[0], 0], -orig_frame[edge[1], 0]
                    z1, z2 = orig_frame[edge[0], 2], orig_frame[edge[1], 2]
                    ax1.plot([x1, x2], [y1, y2], [z1, z2], 'blue', linewidth=2)
            
            ax1.set_title(f'Original F{frame_idx+1}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # マスクデータ
            ax2 = fig.add_subplot(3, 9, frame_idx * 3 + 2, projection='3d')
            mask_frame = mask_sequence[frame_idx]
            mask_x = -mask_frame[:, 1]
            mask_y = -mask_frame[:, 0]
            mask_z = mask_frame[:, 2]
            
            # マスクされていない関節
            unmasked_mask = ~np.isin(np.arange(len(mask_frame)), masked_joints)
            ax2.scatter(mask_x[unmasked_mask], mask_y[unmasked_mask], mask_z[unmasked_mask], 
                       c='blue', s=30, alpha=0.8, label='Unmasked')
            
            # マスクされた関節
            masked_mask = np.isin(np.arange(len(mask_frame)), masked_joints)
            ax2.scatter(mask_x[masked_mask], mask_y[masked_mask], mask_z[masked_mask], 
                       c='red', s=60, marker='x', label='Masked', linewidth=3)
            
            for edge in skeleton_edges:
                if edge[0] < len(mask_frame) and edge[1] < len(mask_frame):
                    x1, x2 = -mask_frame[edge[0], 1], -mask_frame[edge[1], 1]
                    y1, y2 = -mask_frame[edge[0], 0], -mask_frame[edge[1], 0]
                    z1, z2 = mask_frame[edge[0], 2], mask_frame[edge[1], 2]
                    ax2.plot([x1, x2], [y1, y2], [z1, z2], 'blue', linewidth=2, alpha=0.7)
            
            ax2.set_title(f'Masked F{frame_idx+1}')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 再構成データ
            ax3 = fig.add_subplot(3, 9, frame_idx * 3 + 3, projection='3d')
            recon_frame = recon_sequence[frame_idx]
            recon_x = -recon_frame[:, 1]
            recon_y = -recon_frame[:, 0]
            recon_z = recon_frame[:, 2]
            
            # 全関節
            ax3.scatter(recon_x, recon_y, recon_z, c='green', s=30, alpha=0.8, label='Reconstructed')
            
            # マスクされた関節を強調
            ax3.scatter(recon_x[masked_joints], recon_y[masked_joints], recon_z[masked_joints], 
                       c='orange', s=60, marker='o', label='Recon masked', alpha=0.9)
            
            for edge in skeleton_edges:
                if edge[0] < len(recon_frame) and edge[1] < len(recon_frame):
                    x1, x2 = -recon_frame[edge[0], 1], -recon_frame[edge[1], 1]
                    y1, y2 = -recon_frame[edge[0], 0], -recon_frame[edge[1], 0]
                    z1, z2 = recon_frame[edge[0], 2], recon_frame[edge[1], 2]
                    ax3.plot([x1, x2], [y1, y2], [z1, z2], 'green', linewidth=2)
            
            ax3.set_title(f'Reconstructed F{frame_idx+1}')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            # 軸の範囲を統一
            all_x = np.concatenate([orig_x, mask_x, recon_x])
            all_y = np.concatenate([orig_y, mask_y, recon_y])
            all_z = np.concatenate([orig_z, mask_z, recon_z])
            
            margin = 0.1
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim([all_x.min()-margin, all_x.max()+margin])
                ax.set_ylim([all_y.min()-margin, all_y.max()+margin])
                ax.set_zlim([all_z.min()-margin, all_z.max()+margin])
    
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
    
    # 学習パラメータ
    lr = config.get('MODEL', {}).get('lr', 0.001)
    max_epochs = config.get('MODEL', {}).get('max_epoch', 100)
    mask_ratio = config.get('MODEL', {}).get('mask_rate', 0.05)
    weight_decay = config.get('MODEL', {}).get('weight_decay', 0.01)
    optimizer_name = config.get('TRAIN', {}).get('optimizer', 'adam')
    loss_fn = config.get('MODEL', {}).get('loss_fn', 'mse')
    beta = config.get('MODEL', {}).get('beta', 2.0)
    feature_dim = config.get('MODEL', {}).get('feature_dim', 64)
    # Feature-space training only (coordinate reconstruction path removed)

    
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
