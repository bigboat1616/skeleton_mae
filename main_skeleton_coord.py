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
from utils import (
    collapse_mask_indices,
    get_frame_mask_indices,
    get_frame_mask_numpy,
    _ensure_long_tensor,
    plot_reconstruction_comparison,
    plot_sequence_reconstruction_comparison,
    plot_sequence_mask_overview,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

PART_GROUPS = {
    "center_upper": [0, 1, 2, 3, 7],
    "center_lower": [11, 12, 13, 14, 15],
    "right_limb": [4, 5, 6, 16, 17, 18],
    "left_limb": [8, 9, 10, 19, 20, 21],
}
PART_GROUPS2 = {
    "head": [0, 1],
    "neck": [2, 3, 7],
    "upper_torso": [11, 12],
    "lower_torso": [13, 14],
    "hips": [15, 16, 19],
    "right_arm": [4, 5, 6],
    "left_arm": [8, 9, 10],
    "right_leg": [17, 18],
    "left_leg": [20, 21],
}

def sample_bodypart_mask(target=11, tol=1):
    """
    PART_GROUPS2 からランダムにパートを選び、
    関節の合計数が target±tol に収まったところで止める
    """
    part_names = list(PART_GROUPS2.keys())
    random.shuffle(part_names)

    selected_parts = []
    selected_joints = set()

    for name in part_names:
        # このパーツを追加したら何個になるか
        new_joints = selected_joints.union(PART_GROUPS2[name])
        if len(new_joints) == 10 or len(new_joints) ==12:
            continue

        selected_parts.append(name)
        selected_joints = new_joints

        if 11 == len(selected_joints): # 目標に達したら止める
            break

    return sorted(list(selected_joints))

def sample_mask_ratio(mask_ratio=0.5):
    """
    マスクする関節の割合を0.5が平均の正規分布からサンプリングし、0.0から0.9の範囲にクリップする
    """
    return np.clip(np.random.normal(mask_ratio, 0.3), 0.0, 0.9)

def mask_skeleton_joints(data, mask_ratio=0.15, mask_token=0.0, mask_indices=None, batch_idx=0):
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
    
    if mask_indices is None:
        mask_indices = []
        for _ in range(batch_size):
            per_frame = []
            # if batch_idx%2==0:
            # mask_ratio = sample_mask_ratio()
            num_masked = int(num_joints * mask_ratio)
            # masked_joints = torch.randperm(num_joints, device=device)[:num_masked]
            for _ in range(seq_len):
                # 各部位ごとにマスクする関節数を計算
                # if batch_idx%2==1:
                masked_joints = torch.randperm(num_joints, device=device)[:num_masked]
                per_frame.append(masked_joints)
                # パーツまとめてマスク
                # chosen = random.sample(list(PART_GROUPS.keys()), 2)
                # masked_joints = sorted(set(idx for name in chosen for idx in PART_GROUPS[name]))
                # per_frame.append(torch.tensor(masked_joints, device=device, dtype=torch.long))
                # 細かいパーツまとめてマスク
                # target_joints = sample_bodypart_mask(target=11, tol=1)
                # per_frame.append(torch.tensor(target_joints, device=device, dtype=torch.long))
            mask_indices.append(per_frame)

    if isinstance(mask_token, torch.Tensor):
        token = mask_token.to(device)
        if token.dim() == 4:
            token = token.squeeze(0).squeeze(0).squeeze(0)
    else:
        token = mask_token

    for b in range(batch_size):
        for t in range(seq_len):
            masked_joints = mask_indices[b][t]
            if masked_joints.numel() == 0:
                continue
            if isinstance(token, torch.Tensor):
                masked_data[b, t, masked_joints, :] = token
            else:
                masked_data[b, t, masked_joints, :] = token
    
    return masked_data, mask_indices


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
    batch_size, seq_len, num_joints, _ = original.shape
    device = original.device
    
    # 損失関数の選択
    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = nn.MSELoss(reduction='none')
    
    # 全関節の損失を計算
    all_losses = loss_fn(reconstructed, original)  # [batch_size, seq_len, num_joints, 3]
    # フレーム単位で集計
    masked_loss_total = torch.tensor(0.0, device=device)
    masked_elem_count = 0
    unmasked_loss_total = torch.tensor(0.0, device=device)
    unmasked_elem_count = 0
    masked_joint_tally = 0
    unmasked_joint_tally = 0
    
    total_frames = batch_size * seq_len
    
    for b in range(batch_size):
        for t in range(seq_len):
            if mask_indices is None:
                raw_mask = torch.empty(0, dtype=torch.long, device=device)
            else:
                try:
                    raw_mask = mask_indices[b][t]
                except (TypeError, IndexError):
                    raw_mask = get_frame_mask_indices(mask_indices, sample_idx=b, frame_idx=t)
            frame_mask = _ensure_long_tensor(raw_mask, device=device)
            if frame_mask.numel() > 0:
                frame_mask = torch.unique(frame_mask)
            
            mask_flags = torch.zeros(num_joints, dtype=torch.bool, device=device)
            if frame_mask.numel() > 0:
                mask_flags[frame_mask] = True
                masked_slice = all_losses[b, t, frame_mask, :]
                masked_loss_total = masked_loss_total + masked_slice.sum()
                masked_elem_count += masked_slice.numel()
                masked_joint_tally += frame_mask.numel()
            
            frame_unmasked = torch.arange(num_joints, device=device)[~mask_flags]
            if frame_unmasked.numel() > 0:
                unmasked_slice = all_losses[b, t, frame_unmasked, :]
                unmasked_loss_total = unmasked_loss_total + unmasked_slice.sum()
                unmasked_elem_count += unmasked_slice.numel()
                unmasked_joint_tally += frame_unmasked.numel()
    
    # 平均損失の計算（該当要素数で正規化）
    masked_avg_loss = masked_loss_total / masked_elem_count if masked_elem_count > 0 else torch.tensor(0.0, device=device)
    unmasked_avg_loss = unmasked_loss_total / unmasked_elem_count if unmasked_elem_count > 0 else torch.tensor(0.0, device=device)
    
    # 総損失
    # total_loss = masked_avg_loss + unmasked_avg_loss
    total_loss = all_losses.mean()


    # 統計情報
    avg_masked_joints = (masked_joint_tally / total_frames) if total_frames > 0 else 0.0
    avg_unmasked_joints = (unmasked_joint_tally / total_frames) if total_frames > 0 else 0.0
    
    # 関節あたりの損失（正規化）
    total_loss_per_joint = total_loss.item() / original.shape[2]
    masked_loss_per_joint = masked_avg_loss.item() / avg_masked_joints if avg_masked_joints > 0 else 0.0
    unmasked_loss_per_joint = unmasked_avg_loss.item() / avg_unmasked_joints if avg_unmasked_joints > 0 else 0.0
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'total_loss_per_joint': total_loss_per_joint,
        'masked_joints_loss_mean': masked_avg_loss.item(),
        'masked_joints_loss_std': 0.0,  # 簡略化
        'masked_joints_loss_per_joint': masked_loss_per_joint,
        'unmasked_joints_loss_mean': unmasked_avg_loss.item(),
        'unmasked_joints_loss_std': 0.0,  # 簡略化
        'unmasked_joints_loss_per_joint': unmasked_loss_per_joint,
        'num_masked_joints': avg_masked_joints,
        'num_unmasked_joints': avg_unmasked_joints,
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
            data_bn=True,
            layer_num = 3
        )

        self.coord_decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_channels),
        )

    def forward(self, x):
        batch_size, seq_len, num_joints, _ = x.shape # [N, T, V, C]
    
        encoded = self.encoder(x.permute(0, 3, 1, 2)) # [N, C, T, V] 
        encoded = encoded.permute(0, 2, 3, 1)  # [N, T, V, C]
        decoded = self.coord_decoder(encoded.reshape(-1, self.feature_dim)) # [N*T*V, C]
        reconstructed = decoded.view(batch_size, seq_len, num_joints, self.out_channels) # [N, T, V, 3]

        return reconstructed

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
    fixed_union = [collapse_mask_indices(mask_idx) for mask_idx in fixed_mask_indices]
    print(f"Fixed batch for reconstruction tracking: {fixed_batch.shape}")
    print(f"Fixed masked joint indices: {fixed_mask_indices}")
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
                batch, mask_ratio=mask_ratio, mask_token=model.mask_token, batch_idx=batch_idx
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
        plot_sequence_mask_overview(
            test_batch.cpu(),
            test_mask_indices,
            save_path=f"{save_dir}/sequence_mask_overview.png",
            sample_idx=0,
        )
    
    return model, training_history


class SkeletonDataset(Dataset):
    """JTA-3DPデータセット用のPyTorch Dataset"""
    
    def __init__(self, json_files, track_size=9, sequence_length=9, num_joints=22, frequency=1):
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
    save_dir = "ckpt/sampling"  # checkpoint保存ディレクトリ
    
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
