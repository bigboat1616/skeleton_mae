"""
Shared utilities for skeleton-based experiments.

This module hosts the dataset loader, masking helper, and filesystem helpers
that are required by both `main_skeleton.py` and `main_skeleton_coord.py`.
Keeping them in a single place avoids duplication and ensures consistent
behaviour across the different training scripts.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MaskIndex = Union[int, torch.Tensor]

__all__ = [
    "mask_skeleton_joints",
    "SkeletonDataset",
    "load_json_files",
    "create_visualization_directory",
]


def mask_skeleton_joints(
    data: torch.Tensor,
    mask_ratio: float = 0.15,
    mask_token: Union[float, torch.Tensor] = 0.0,
    mask_indices: Optional[Sequence[Iterable[int]]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Mask skeleton joints along the joint axis.

    Args:
        data: Tensor shaped [batch_size, seq_len, num_joints, channels]
              (channels = 3 for coordinates, feature_dim for features).
        mask_ratio: Ratio of joints to mask when indices are not provided.
        mask_token: Value that replaces the masked joints. Can be a scalar or
                    a tensor that matches the joint feature dimension.
        mask_indices: Optional precomputed indices to mask for each sample.

    Returns:
        masked_data: Masked tensor (same shape as `data`).
        mask_indices: List of tensors with masked joint indices per sample.
    """
    batch_size, _, num_joints, _ = data.shape
    masked_data = data.clone()
    device = data.device

    if mask_indices is None:
        mask_indices = []
        for _ in range(batch_size):
            num_masked = int(num_joints * mask_ratio)
            masked_joints = torch.randperm(num_joints, device=device)[:num_masked]
            mask_indices.append(masked_joints)

    result_indices: List[torch.Tensor] = []
    for b in range(batch_size):
        masked_joints = mask_indices[b]
        if not isinstance(masked_joints, torch.Tensor):
            masked_joints = torch.as_tensor(masked_joints, device=device)
        result_indices.append(masked_joints)

        if masked_joints.numel() == 0:
            continue

        if isinstance(mask_token, torch.Tensor):
            if mask_token.dim() == 4:
                token = mask_token.squeeze(0).squeeze(0).squeeze(0)
            else:
                token = mask_token
            masked_data[b, :, masked_joints, :] = token
        else:
            masked_data[b, :, masked_joints, :] = mask_token

    return masked_data, result_indices


class SkeletonDataset(Dataset):
    """Dataset wrapper for the JTA-3DP skeleton dataset."""

    def __init__(
        self,
        json_files: Sequence[str],
        track_size: int = 16,
        sequence_length: int = 9,
        num_joints: int = 22,
        frequency: int = 1,
    ) -> None:
        self.json_files = list(json_files)
        self.track_size = track_size
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.frequency = frequency
        self.data: List[np.ndarray] = []
        self.file_stats: List[dict] = []

        print(f"Loading {len(self.json_files)} JSON files...")
        for json_file in tqdm(self.json_files):
            patches_before = len(self.data)
            self.load_jta_data_from_json(json_file)
            patches_after = len(self.data)
            self.file_stats.append(
                {
                    "file": os.path.basename(json_file),
                    "patches_added": patches_after - patches_before,
                    "total_patches": patches_after,
                }
            )

        print(f"Total patches loaded: {len(self.data)}")

    def load_jta_data_from_json(self, json_file: str) -> None:
        """Load a single JTA-3DP JSON file and append valid tracks."""
        with open(json_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print(f"Invalid data format in {json_file}")
            return

        frames_data = []
        for frame_data in data:
            if not isinstance(frame_data, list) or len(frame_data) < 4 + self.num_joints * 3:
                print(f"Invalid frame data format in {json_file}")
                continue

            frame_id = frame_data[0]
            person_id = frame_data[1]

            coords_start = 4
            x_coords = frame_data[coords_start : coords_start + self.num_joints]
            y_coords = frame_data[coords_start + self.num_joints : coords_start + self.num_joints * 2]
            z_coords = frame_data[
                coords_start + self.num_joints * 2 : coords_start + self.num_joints * 3
            ]

            joints_3d = np.array([x_coords, y_coords, z_coords]).T

            if not np.allclose(joints_3d, 0) and not np.isnan(joints_3d).any():
                frames_data.append(
                    {
                        "frame_id": frame_id,
                        "person_id": person_id,
                        "joints": joints_3d,
                    }
                )

        if frames_data:
            self._create_tracks_from_frames(frames_data)
        else:
            print(f"No valid frames found in {json_file}")

    def _create_tracks_from_frames(self, frames_data: List[dict]) -> None:
        """Group frames by person id and build track sequences."""
        person_dict = {}
        for frame in frames_data:
            person_dict.setdefault(frame["person_id"], []).append(frame)

        for person_id, person_frames in person_dict.items():
            if len(person_frames) < self.track_size * self.frequency:
                continue

            person_frames.sort(key=lambda x: x["frame_id"])

            for start_idx in range(0, len(person_frames) - self.track_size * self.frequency + 1, self.track_size):
                end_idx = start_idx + self.track_size * self.frequency
                sampled_frames = person_frames[start_idx:end_idx:self.frequency]

                if len(sampled_frames) < self.sequence_length:
                    continue

                joints_sequence = [frame["joints"] for frame in sampled_frames[: self.sequence_length]]
                if len(joints_sequence) == self.sequence_length:
                    patch = np.array(joints_sequence)
                    self.data.append(patch)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.data[idx]
        return torch.tensor(patch, dtype=torch.float32)


def load_json_files(data_dir: str) -> List[str]:
    """Collect JSON files from a directory tree."""
    json_files: List[str] = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
    else:
        print(f"Data directory {data_dir} not found. Using dummy data.")
    return json_files


def create_visualization_directory(path: str = "skeleton/train") -> None:
    """Ensure that the visualization directory exists."""
    os.makedirs(path, exist_ok=True)
    print(f"Created visualization directory: {path}")

