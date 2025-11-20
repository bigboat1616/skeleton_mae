
def plot_skeleton_3d(skeleton_data, title="Skeleton", save_path=None):
    """
    3Dスケルトンをプロット
    
    Args:
        skeleton_data: [num_joints, 3] のスケルトンデータ
        title: プロットのタイトル
        save_path: 保存パス（Noneの場合は表示のみ）
        
    Joint indices:
        0: head_top, 1: head_center, 2: neck, 3: right_clavicle, 4: right_shoulder,
        5: right_elbow, 6: right_wrist, 7: left_clavicle, 8: left_shoulder, 9: left_elbow,
        10: left_wrist, 11: spine0, 12: spine1, 13: spine2, 14: spine3, 15: spine4,
        16: right_hip, 17: right_knee, 18: right_ankle, 19: left_hip, 20: left_knee, 21: left_ankle
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # スケルトンの骨格接続（22関節構造）
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
    
    # 座標系変換: X=右方向+, Y=下方向+, Z=奥方向+ (Z軸を反時計回りに90度回転)
    x = -skeleton_data[:, 1]  # X: 右方向+ (元データのy座標を反転してX軸に)
    y = -skeleton_data[:, 0]  # Y: 下方向+ (元データのx座標を反転してY軸に)
    z = skeleton_data[:, 2]   # Z: 奥方向+ (元データのz座標)
    
    # 関節をプロット
    ax.scatter(x, y, z, c='red', s=50)
    
    # 骨格をプロット
    for edge in skeleton_edges:
        if edge[0] < len(skeleton_data) and edge[1] < len(skeleton_data):
            # 座標変換を適用
            x1, x2 = -skeleton_data[edge[0], 1], -skeleton_data[edge[1], 1]
            y1, y2 = -skeleton_data[edge[0], 0], -skeleton_data[edge[1], 0]
            z1, z2 = skeleton_data[edge[0], 2], skeleton_data[edge[1], 2]
            
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2)
    
    ax.set_xlabel('X+')
    ax.set_ylabel('Y+')
    ax.set_zlabel('Z+')
    ax.set_title(title)
    
    # 視点をX-Y平面に対して90度（垂直）になるように設定
    ax.view_init(elev=90, azim=180)  # elev=90: 垂直方向から, azim=180: X-Y平面に対して垂直
    
    if save_path:
        plt.savefig(save_path)
        print(f"Skeleton plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_patch_sequence(patches, patch_idx=0, save_path=None):
    """
    パッチのシーケンスをアニメーションでプロット
    
    Args:
        patches: [num_patches, sequence_length, num_joints, 3] のパッチデータ
        patch_idx: 表示するパッチのインデックス
        save_path: 保存パス（Noneの場合は表示のみ）
    """
    if patch_idx >= len(patches):
        print(f"Invalid patch_idx: {patch_idx}. Available patches: {len(patches)}")
        return
    
    patch = patches[patch_idx]  # [sequence_length, num_joints, 3]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        skeleton_data = patch[frame]  # [num_joints, 3]
        
        # 座標系変換: X=右方向+, Y=下方向+, Z=奥方向+ (Z軸を反時計回りに90度回転)
        x = -skeleton_data[:, 1]  # X: 右方向+ (元データのy座標を反転してX軸に)
        y = -skeleton_data[:, 0]  # Y: 下方向+ (元データのx座標を反転してY軸に)
        z = skeleton_data[:, 2]   # Z: 奥方向+ (元データのz座標)
        
        # 関節をプロット
        ax.scatter(x, y, z, c='red', s=50)
        
        # 骨格をプロット
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
        
        for edge in skeleton_edges:
            if edge[0] < len(skeleton_data) and edge[1] < len(skeleton_data):
                # 座標変換を適用
                x1, x2 = -skeleton_data[edge[0], 1], -skeleton_data[edge[1], 1]
                y1, y2 = -skeleton_data[edge[0], 0], -skeleton_data[edge[1], 0]
                z1, z2 = skeleton_data[edge[0], 2], skeleton_data[edge[1], 2]
                
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2)
        
        ax.set_xlabel('X+')
        ax.set_ylabel('Y+')
        ax.set_zlabel('Z+')
        ax.set_title(f'Patch {patch_idx} - Frame {frame}/{len(patch)-1}')
        
        # 視点をX-Y平面に対して90度（垂直）になるように設定
        ax.view_init(elev=90, azim=180)  # elev=90: 垂直方向から, azim=180: X-Y平面に対して垂直
        
        # 軸の範囲を固定（変換後の座標を使用）
        ax.set_xlim([x.min()-0.5, x.max()+0.5])
        ax.set_ylim([y.min()-0.5, y.max()+0.5])
        ax.set_zlim([z.min()-0.5, z.max()+0.5])
    
    anim = animation.FuncAnimation(fig, animate, frames=len(patch), interval=200, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_all_frames_static(skeleton_patch, save_path, title="All Frames Skeleton"):
    """
    スケルトンの全フレームを一つの静止画像に表示
    
    Args:
        skeleton_patch: [sequence_length, num_joints, 3] のスケルトンパッチデータ
        save_path: 保存パス
        title: プロットのタイトル
    """
    sequence_length = skeleton_patch.shape[0]
    
    # 単一の3Dプロットを作成
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # スケルトンの骨格接続
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
    
    # 全フレームの座標範囲を計算
    all_x = []
    all_y = []
    all_z = []
    
    # 各フレームのデータを収集
    for frame_idx in range(sequence_length):
        skeleton_data = skeleton_patch[frame_idx]  # [num_joints, 3]
        
        # 座標系変換: X=右方向+, Y=下方向+, Z=奥方向+
        x = -skeleton_data[:, 1]  # X: 右方向+ (元データのy座標を反転してX軸に)
        y = -skeleton_data[:, 0]  # Y: 下方向+ (元データのx座標を反転してY軸に)
        z = skeleton_data[:, 2]   # Z: 奥方向+ (元データのz座標)
        
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
    
    # 色のグラデーション（フレームごとに異なる色）
    colors = plt.cm.viridis(np.linspace(0, 1, sequence_length))
    
    # 各フレームを重ねてプロット
    for frame_idx in range(sequence_length):
        skeleton_data = skeleton_patch[frame_idx]  # [num_joints, 3]
        
        # 座標系変換
        x = -skeleton_data[:, 1]
        y = -skeleton_data[:, 0]
        z = skeleton_data[:, 2]
        
        # 関節をプロット（フレームごとに異なる色）
        ax.scatter(x, y, z, c=[colors[frame_idx]], s=50, alpha=0.7, label=f'Frame {frame_idx}')
        
        # 骨格をプロット
        for edge in skeleton_edges:
            if edge[0] < len(skeleton_data) and edge[1] < len(skeleton_data):
                # 座標変換を適用
                x1, x2 = -skeleton_data[edge[0], 1], -skeleton_data[edge[1], 1]
                y1, y2 = -skeleton_data[edge[0], 0], -skeleton_data[edge[1], 0]
                z1, z2 = skeleton_data[edge[0], 2], skeleton_data[edge[1], 2]
                
                ax.plot([x1, x2], [y1, y2], [z1, z2], color=colors[frame_idx], linewidth=2, alpha=0.7)
    
    # 軸設定
    ax.set_xlabel('X+')
    ax.set_ylabel('Y+')
    ax.set_zlabel('Z+')
    ax.set_title(title)
    
    # 視点設定
    ax.view_init(elev=90, azim=180)
    
    # 軸の範囲を固定（全フレームの範囲を使用）
    ax.set_xlim([min(all_x)-0.5, max(all_x)+0.5])
    ax.set_ylim([min(all_y)-0.5, max(all_y)+0.5])
    ax.set_zlim([min(all_z)-0.5, max(all_z)+0.5])
    
    # 凡例を追加
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"All frames skeleton plot saved to {save_path}")
    plt.close()
