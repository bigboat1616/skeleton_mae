import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from main_skeleton_coord import (
    STGCN18Reconstructor,
    SkeletonDataset,
    mask_skeleton_joints,
)


def parse_mask_ratios(ratios: str) -> List[float]:
    values = sorted({float(v.strip()) for v in ratios.split(",") if v.strip()})
    if not values:
        raise ValueError("mask_ratios must contain at least one value")
    if values[0] != 0.0:
        values.insert(0, 0.0)
    return values


def build_val_dataloader(
    config: Dict,
    batch_size: int,
    num_workers: int,
    subset: int = None,
) -> DataLoader:
    input_dir = config.get("DATA", {}).get("input_dir", "data/jta_3dp_row")
    data_dir = os.path.join(input_dir, "val")
    track_size = config.get("TRAIN", {}).get("track_size", 9)
    sequence_length = config.get("TRAIN", {}).get("input_track_size", 9)
    num_joints = config.get("MODEL", {}).get("num_joints", 22)
    frequency = config.get("TRAIN", {}).get("frequency", 1)

    json_files = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".json"):
                    json_files.append(os.path.join(root, filename))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    if subset is not None and subset > 0:
        json_files = json_files[:subset]

    dataset = SkeletonDataset(
        json_files=json_files,
        track_size=track_size,
        sequence_length=sequence_length,
        num_joints=num_joints,
        frequency=frequency,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def extract_encoder_features(model: STGCN18Reconstructor, batch: torch.Tensor) -> torch.Tensor:
    # batch: [B, T, V, C]
    data = batch.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
    features, _ = model.encoder.extract_feature(data)
    return features.permute(0, 2, 3, 1).contiguous()  # [B, T, V, feature_dim]


def flatten_features(features: torch.Tensor) -> torch.Tensor:
    return features.view(features.size(0), -1)


def evaluate_mask_robustness(
    model: STGCN18Reconstructor,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratios: Iterable[float],
    mask_token_mode: str,
    max_batches: int,
) -> Tuple[Dict[float, List[float]], Dict[float, Dict[str, List[float]]], Dict[float, List[int]]]:
    cosine_scores: Dict[float, List[float]] = {ratio: [] for ratio in mask_ratios}
    distance_scores: Dict[float, Dict[str, List[float]]] = {
        ratio: {"l2": [], "l1": []} for ratio in mask_ratios if ratio > 0
    }
    masked_joint_counts: Dict[float, List[int]] = {ratio: [] for ratio in mask_ratios if ratio > 0}

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if 0 < max_batches <= batch_idx:
                break

            batch = batch.to(device)

            baseline_features = extract_encoder_features(model, batch)
            baseline_flat = flatten_features(baseline_features)

            baseline_flat_norm = F.normalize(baseline_flat, p=2, dim=1)

            for ratio in mask_ratios:
                if ratio == 0.0:
                    masked_flat_norm = baseline_flat_norm
                    cosine = torch.sum(baseline_flat_norm * masked_flat_norm, dim=1)
                    cosine_scores[ratio].extend(cosine.detach().cpu().tolist())
                    continue

                mask_token = model.mask_token if mask_token_mode == "learned" else 0.0
                masked_batch, mask_indices = mask_skeleton_joints(
                    batch,
                    mask_ratio=ratio,
                    mask_token=mask_token,
                )

                masked_features = extract_encoder_features(model, masked_batch)
                masked_flat = flatten_features(masked_features)
                masked_flat_norm = F.normalize(masked_flat, p=2, dim=1)

                cosine = torch.sum(baseline_flat_norm * masked_flat_norm, dim=1)
                cosine_scores[ratio].extend(cosine.detach().cpu().tolist())

                if ratio > 0:
                    l2_dist = torch.norm(baseline_flat - masked_flat, p=2, dim=1)
                    l1_dist = torch.norm(baseline_flat - masked_flat, p=1, dim=1)
                    distance_scores[ratio]["l2"].extend(l2_dist.detach().cpu().tolist())
                    distance_scores[ratio]["l1"].extend(l1_dist.detach().cpu().tolist())

                # count masked joints per sample for reporting
                counts = [
                    int(idx.numel()) if torch.is_tensor(idx) else len(idx)
                    for idx in mask_indices
                ]
                masked_joint_counts[ratio].extend(counts)

    if not cosine_scores[mask_ratios[0]]:
        raise RuntimeError("No samples evaluated; check dataloader/max_batches settings.")

    return cosine_scores, distance_scores, masked_joint_counts


def summarize_metrics(
    cosine_scores: Dict[float, List[float]],
    distance_scores: Dict[float, Dict[str, List[float]]],
    masked_joint_counts: Dict[float, List[int]],
) -> Dict[float, Dict[str, float]]:
    summary = {}
    for ratio, scores in cosine_scores.items():
        scores_array = np.asarray(scores, dtype=np.float64)
        stats = {
            "cosine_mean": float(scores_array.mean()),
            "cosine_std": float(scores_array.std()),
            "cosine_min": float(scores_array.min()),
            "cosine_max": float(scores_array.max()),
            "num_samples": int(scores_array.size),
        }
        if ratio > 0 and ratio in distance_scores:
            l2_array = np.asarray(distance_scores[ratio]["l2"], dtype=np.float64)
            l1_array = np.asarray(distance_scores[ratio]["l1"], dtype=np.float64)
            stats.update(
                {
                    "l2_mean": float(l2_array.mean()),
                    "l2_std": float(l2_array.std()),
                    "l1_mean": float(l1_array.mean()),
                    "l1_std": float(l1_array.std()),
                }
            )
        if ratio > 0 and ratio in masked_joint_counts:
            counts = np.asarray(masked_joint_counts[ratio], dtype=np.float64)
            stats["masked_joints_mean"] = float(counts.mean())
            stats["masked_joints_std"] = float(counts.std())
        summary[ratio] = stats
    return summary


def save_results(path: str, summary: Dict[float, Dict[str, float]], raw_scores: Dict[float, List[float]]) -> None:
    payload = {
        "summary": {str(k): v for k, v in summary.items()},
        "raw_cosine_scores": {str(k): v for k, v in raw_scores.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate encoder robustness against random joint masking.")
    parser.add_argument("--ckpt", required=True, help="Path to encoder checkpoint (.pth)")
    parser.add_argument("--cfg", default="configs_skeleton.yml", help="Config YAML path for dataset parameters")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=1000, help="Number of batches to evaluate (0 for all)")
    parser.add_argument("--mask_ratios", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--mask_token", type=str, choices=["zero", "learned"], default="learned",
                        help="Mask token to use when removing joints.")
    parser.add_argument("--subset_json", type=int, default=None, help="Use only the first N JSON files (debugging).")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results.")
    args = parser.parse_args()

    if os.path.exists(args.cfg):
        with open(args.cfg, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: config {args.cfg} not found. Using defaults.")
        config = {}

    device = torch.device(args.device)
    mask_ratios = parse_mask_ratios(args.mask_ratios)

    feature_dim = config.get("MODEL", {}).get("feature_dim", 256)
    model = STGCN18Reconstructor(
        in_channels=3,
        out_channels=3,
        feature_dim=feature_dim,
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in checkpoint["encoder_state_dict"].items() if k in state_dict})
        model.load_state_dict(state_dict, strict=False)
        print("Loaded encoder_state_dict into STGCN18Reconstructor.")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Loaded full model_state_dict.")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded raw state_dict.")

    dataloader = build_val_dataloader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset_json,
    )

    cosine_scores, distance_scores, masked_counts = evaluate_mask_robustness(
        model=model,
        dataloader=dataloader,
        device=device,
        mask_ratios=mask_ratios,
        mask_token_mode=args.mask_token,
        max_batches=args.max_batches,
    )

    summary = summarize_metrics(cosine_scores, distance_scores, masked_counts)

    print("\n=== Encoder Cosine Similarity under Joint Masking ===")
    for ratio in mask_ratios:
        stats = summary[ratio]
        ratio_pct = ratio * 100
        print(
            f"Mask {ratio_pct:5.1f}% | cos μ={stats['cosine_mean']:.4f}, σ={stats['cosine_std']:.4f}, "
            f"[min={stats['cosine_min']:.4f}, max={stats['cosine_max']:.4f}] "
            f"(n={stats['num_samples']})"
        )
        if ratio > 0 and ratio in masked_counts:
            print(
                f"              avg masked joints/sample = {stats['masked_joints_mean']:.2f} "
                f"(σ={stats['masked_joints_std']:.2f})"
            )
            if ratio in distance_scores:
                print(
                    f"              L2 μ={stats['l2_mean']:.4f}, σ={stats['l2_std']:.4f} | "
                    f"L1 μ={stats['l1_mean']:.4f}, σ={stats['l1_std']:.4f}"
                )

    if args.output:
        payload = {
            "summary": {str(k): v for k, v in summary.items()},
            "raw_cosine_scores": {str(k): v for k, v in cosine_scores.items()},
            "raw_distance_scores": {
                str(k): {metric: vals for metric, vals in metrics.items()}
                for k, metrics in distance_scores.items()
            },
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

