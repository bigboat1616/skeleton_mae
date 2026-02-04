import os
import json
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from main_skeleton_coord import (
    STGCN18Reconstructor,
    SkeletonDataset,
    mask_skeleton_joints,
    compute_reconstruction_loss,
)
from utils import (
    calculate_masked_unmasked_batch_errors,
    plot_reconstruction_comparison,
    plot_sequence_reconstruction_comparison,
    plot_sequence_mask_overview,
)


def set_global_seed(seed: int):
    """Make masking and dataloading reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_encoder_features(model: STGCN18Reconstructor, batch: torch.Tensor) -> torch.Tensor:
    """
    Encode skeleton sequences via ST-GCN encoder.
    Returns tensor of shape [B, T, V, feature_dim].
    """
    data = batch.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
    features = model.encoder(data) # [B, C, T, V]
    return features.permute(0, 2, 3, 1).contiguous()


def load_checkpoint_flex(model, ckpt_path, map_location="cpu"):
    """
    様々な形式のチェックポイントを読み込み:
      - {'model_state_dict': ...}
      - {'encoder_state_dict': ...}  (encoderのみ)
      - 直接 state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        # missing, unexpected = model.load_state_dict(state, strict=False)
        # print(
        #     f"Loaded full model_state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        # )
        # if missing:
        #     print(f"  Missing keys: {missing}")
        # if unexpected:
        #     print(f"  Unexpected keys: {unexpected}")
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        enc_state = ckpt["encoder_state_dict"]
        model_state = model.state_dict()
        filtered = {k: v for k, v in enc_state.items() if k in model_state}
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        # print(f"Loaded encoder_state_dict into model. Copied params: {len(filtered)}")
        missing = [k for k in model_state.keys() if k not in filtered]
        # if missing:
        #     print(f"  Remaining params will use initialization: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        # print(
        #     f"Loaded raw state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        # )
        # if missing:
        #     print(f"  Missing keys: {missing}")
        # if unexpected:
        #     print(f"  Unexpected keys: {unexpected}")


def build_val_dataloader(config, batch_size=4, num_workers=0):
    input_dir = config.get("DATA", {}).get("input_dir", "data/jta_3dp")
    data_dir = os.path.join(input_dir, "test")
    track_size = config.get("TRAIN", {}).get("track_size", 9)
    sequence_length = config.get("TRAIN", {}).get("input_track_size", 9)
    num_joints = 22
    frequency = config.get("TRAIN", {}).get("frequency", 1)

    json_files = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith(".json"):
                    json_files.append(os.path.join(root, filename))
    # print(f"VAL json files: {len(json_files)} in {data_dir}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    dataset = SkeletonDataset(
        json_files=json_files,
        track_size=track_size,
        sequence_length=sequence_length,
        num_joints=num_joints,
        frequency=frequency,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def replace_3d_coordinates_with_reconstructions(
    dataset: SkeletonDataset,
    reconstructions: torch.Tensor,
    mask_indices_per_sample,
    input_root: str,
    output_root: str,
) -> None:
    """
    Replace the 3D joint coordinates in source JSON files with reconstructed values.
    """
    if reconstructions.size(0) != len(dataset):
        raise ValueError("Reconstruction count does not match dataset length.")
    if len(mask_indices_per_sample) != len(dataset):
        raise ValueError("Mask indices count does not match dataset length.")

    os.makedirs(output_root, exist_ok=True)

    reconstructions = reconstructions.detach().cpu()

    # Load JSON files once and keep them cached for updates
    file_cache = {}
    for json_path in dataset.json_files:
        with open(json_path, "r") as handle:
            file_cache[json_path] = json.load(handle)

    num_joints = dataset.num_joints

    for sample_idx in range(len(dataset)):
        metadata = dataset.get_metadata(sample_idx)
        source_file = metadata["source_file"]
        raw_indices = metadata["raw_indices"]
        predictions = reconstructions[sample_idx]
        sample_masks = mask_indices_per_sample[sample_idx]

        if len(raw_indices) != predictions.size(0):
            raise ValueError(
                f"Sequence length mismatch for sample {sample_idx}: "
                f"metadata={len(raw_indices)}, recon={predictions.size(0)}"
            )

        frames = file_cache[source_file]
        for frame_step, (frame_pos, joints) in enumerate(zip(raw_indices, predictions)):
            if frame_pos >= len(frames):
                continue

            frame = frames[frame_pos]
            if len(frame) < 4 + num_joints * 3:
                continue

            joints_np = joints.detach().cpu().numpy()
            if frame_step >= len(sample_masks):
                mask_joint_indices = []
            else:
                mask_entry = sample_masks[frame_step]
                if torch.is_tensor(mask_entry):
                    mask_joint_indices = mask_entry.tolist()
                else:
                    mask_joint_indices = list(mask_entry)
            normalized_indices = []
            for joint_idx in mask_joint_indices:
                if hasattr(joint_idx, "item"):
                    normalized_indices.append(int(joint_idx.item()))
                else:
                    normalized_indices.append(int(joint_idx))
            mask_joint_indices = normalized_indices

            if not mask_joint_indices:
                continue

            start = 4
            for joint_idx in mask_joint_indices:
                if joint_idx < 0 or joint_idx >= num_joints:
                    continue
                frame[start + joint_idx] = float(joints_np[joint_idx, 0])
                frame[start + num_joints + joint_idx] = float(joints_np[joint_idx, 1])
                frame[start + 2 * num_joints + joint_idx] = float(joints_np[joint_idx, 2])

    for source_path, frames in file_cache.items():
        relative_path = os.path.relpath(source_path, input_root)
        destination = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "w") as handle:
            json.dump(frames, handle)


def _to_cpu_mask_indices(mask_indices):
    cpu_indices = []
    for sample_indices in mask_indices:
        per_frame = []
        for frame_indices in sample_indices:
            if torch.is_tensor(frame_indices):
                per_frame.append(frame_indices.detach().cpu())
            else:
                per_frame.append(torch.tensor(frame_indices))
        cpu_indices.append(per_frame)
    return cpu_indices


def evaluate_coordinate_reconstruction(
    model,
    dataloader,
    device,
    mask_ratio=0.5,
    max_batches=10000,
    save_dir=None,
    loss_type="mse",
    beta=2.0,
    log_every=1,
):
    model.eval()

    total_losses = []
    masked_losses = []
    per_batch_stats = []

    originals = []
    reconstructions = []
    all_mask_indices = []
    cosine_sims = []

    best_sample_error = None
    best_sample_info = None

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            batch = batch.to(device)

            masked_batch, mask_indices = mask_skeleton_joints(
                batch, mask_ratio=mask_ratio, mask_token=model.mask_token
            )

            # Clean vs masked encoder features

            reconstructed = model(masked_batch)
            masked_features = model.encoder(masked_batch.permute(0, 3, 1, 2).contiguous())
            clean_features = model.encoder(batch.permute(0, 3, 1, 2).contiguous())

            loss, loss_dict = compute_reconstruction_loss(
                batch,
                reconstructed,
                clean_features,
                masked_features,
                mask_indices,
                loss_type=loss_type,
                beta=beta,
            )

            distances = torch.norm(batch - reconstructed, dim=-1)  # [B, T, V]
            sample_errors = distances.sum(dim=(1, 2)).detach().cpu()
            batch_min_error, batch_min_idx = torch.min(sample_errors, dim=0)
            if best_sample_error is None or batch_min_error.item() < best_sample_error:
                best_sample_error = batch_min_error.item()
                best_sample_info = {
                    "batch_idx": batch_idx,
                    "sample_idx": batch_min_idx.item(),
                    "error": best_sample_error,
                }

            total_losses.append(loss.item())
            masked_losses.append(loss_dict["masked_joints_loss_mean"])
            per_batch_stats.append(loss_dict)

            originals.append(batch.detach().cpu())
            reconstructions.append(reconstructed.detach().cpu())
            all_mask_indices.extend(_to_cpu_mask_indices(mask_indices))

            clean_flat = clean_features.detach().reshape(clean_features.size(0), -1)
            masked_flat = masked_features.detach().reshape(masked_features.size(0), -1)
            cosine = F.cosine_similarity(clean_flat, masked_flat, dim=1)
            cosine_sims.append(cosine.cpu())

            if save_dir is not None:
                batch_dir = os.path.join(save_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)

                # plot_reconstruction_comparison(
                #     batch.detach().cpu(),
                #     masked_batch.detach().cpu(),
                #     reconstructed.detach().cpu(),
                #     mask_indices,
                #     os.path.join(batch_dir, "comparison.png"),
                #     overlay=False,
                # )

                # plot_reconstruction_comparison(
                #     batch.detach().cpu(),
                #     masked_batch.detach().cpu(),
                #     reconstructed.detach().cpu(),
                #     mask_indices,
                #     os.path.join(batch_dir, "comparison_overlay.png"),
                #     overlay=True,
                # )
                # plot_sequence_reconstruction_comparison(
                #         batch.detach().cpu(),
                #         masked_batch.detach().cpu(),
                #         reconstructed.detach().cpu(),
                #         mask_indices,
                #         os.path.join(batch_dir, "sequence.png"),
                #         overlay=False,
                #     )
                # plot_sequence_mask_overview(
                #         batch.detach().cpu(),
                #         mask_indices,
                #         os.path.join(batch_dir, "mask_overview.png"),
                #     )

                # plot_sequence_reconstruction_comparison(
                #     batch.detach().cpu(),
                #     masked_batch.detach().cpu(),
                #     reconstructed.detach().cpu(),
                #     mask_indices,
                #     os.path.join(batch_dir, "sequence_overlay.png"),
                #     overlay=True,
                # )

            # if (batch_idx + 1) % log_every == 0:
            #     print(
            #         f"[Batch {batch_idx + 1}] "
            #         f"loss={loss.item():.6f}, masked={loss_dict['masked_joints_loss_mean']:.6f}"
            #     )

    if len(originals) == 0:
        raise RuntimeError("No batches evaluated.")

    if best_sample_info is not None:
        print(
            "Lowest-error sample → batch {batch_idx}, sample {sample_idx}, total distance {error:.6f} m".format(
                batch_idx=best_sample_info["batch_idx"],
                sample_idx=best_sample_info["sample_idx"],
                error=best_sample_info["error"],
            )
        )

    original_cat = torch.cat(originals, dim=0)
    recon_cat = torch.cat(reconstructions, dim=0)

    stats = calculate_masked_unmasked_batch_errors(
        original_cat, recon_cat, all_mask_indices
    )

    cosine_cat = torch.cat(cosine_sims, dim=0) if cosine_sims else torch.tensor([])

    summary = {
        "per_batch": per_batch_stats,
        **stats,
    }

    summary["cosine_similarity_mean"] = float(cosine_cat.mean().item()) if cosine_sims else 0.0
    summary["cosine_similarity_std"] = float(cosine_cat.std(unbiased=False).item()) if cosine_cat.numel() > 0 else 0.0
    summary["cosine_similarity_values"] = cosine_cat.tolist() if cosine_sims else []

    return summary, recon_cat, all_mask_indices


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--cfg", type=str, default="configs_skeleton.yml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=1000)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1", "rce"])
    parser.add_argument("--beta", type=float, default=2.0, help="RCE beta (used when loss_type='rce')")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="skeleton/val_coord_visualizations",
        help="Directory to save visualization images",
    )
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for joint masking reproducibility")
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Optional path to append textual summary of metrics.",
    )
    args = parser.parse_args()

    if os.path.exists(args.cfg):
        with open(args.cfg, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    input_root = config.get("DATA", {}).get("input_dir", "data/jta_3dp_row")
    test_input_dir = os.path.join(input_root, "test")
    cleaned_output_dir = os.path.join(input_root, "test_clean")

    device = torch.device(args.device)
    set_global_seed(args.seed)

    feature_dim = config.get("MODEL", {}).get("feature_dim", 256)
    model = STGCN18Reconstructor(
        in_channels=3,
        out_channels=3,
        feature_dim=feature_dim,
    ).to(device)
    encoder_total_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable_params = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    print(
        "Encoder parameter count → total: {:,} (trainable: {:,})".format(
            encoder_total_params, encoder_trainable_params
        )
    )

    load_checkpoint_flex(model, args.ckpt, map_location=device)

    print("################",model.mask_token,"################")

    dataloader = build_val_dataloader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # print(f"\nSaving visualizations to: {args.save_dir}")
    stats, recon_sequences, mask_indices_per_sample = evaluate_coordinate_reconstruction(
        model,
        dataloader,
        device,
        mask_ratio=args.mask_ratio,
        max_batches=args.max_batches,
        save_dir=args.save_dir,
        loss_type=args.loss_type,
        beta=args.beta,
        log_every=args.log_every,
    )

    replace_3d_coordinates_with_reconstructions(
        dataloader.dataset,
        recon_sequences,
        mask_indices_per_sample,
        test_input_dir,
        cleaned_output_dir,
    )
    print(f"Saved reconstructed JSON files to {cleaned_output_dir}")

    summary_lines = [
        "=== Coordinate Reconstruction on VAL ===",
        f"Mask ratio in training:    {args.ckpt}",
        f"Mask ratio in validation: {args.mask_ratio}",
        f"Masked micro mean:      {stats['micro_masked_mean']:.6f} m",
        f"Unmasked micro mean:    {stats['micro_unmasked_mean']:.6f} m",
        # f"Masked macro mean:      {stats['macro_masked_mean']:.6f} m",
        # f"Unmasked macro mean:    {stats['macro_unmasked_mean']:.6f} m",
        # f"Masked instances:       {stats['total_masked_instances']}",
        # f"Unmasked instances:     {stats['total_unmasked_instances']}",
        # f"Avg masked/sample:      {stats['avg_masked_per_sample']:.2f} ({stats['mask_rate']*100:.1f}%)",
        f"Average micro mean:    {(stats['micro_masked_mean']*stats['total_masked_instances']+stats['micro_unmasked_mean']*stats['total_unmasked_instances'])/(stats['total_masked_instances']+stats['total_unmasked_instances']):.6f} m",
        f"Cosine similarity mean: {stats['cosine_similarity_mean']:.6f}",
        f"Cosine similarity std:  {stats['cosine_similarity_std']:.6f}",
    ]

    print()
    for line in summary_lines:
        print(line)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] checkpoint={args.ckpt}, mask_ratio={args.mask_ratio}\n")
            for line in summary_lines:
                f.write(line + "\n")
            f.write("\n")


if __name__ == "__main__":
    main()

