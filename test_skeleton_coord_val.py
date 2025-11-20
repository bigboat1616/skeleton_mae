import os
import yaml
import torch
from torch.utils.data import DataLoader

from main_skeleton_coord import (
    STGCN18Reconstructor,
    SkeletonDataset,
    mask_skeleton_joints,
    compute_reconstruction_loss,
    calculate_masked_unmasked_batch_errors,
    plot_reconstruction_comparison,
    plot_sequence_reconstruction_comparison,
)


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
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded full model_state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        enc_state = ckpt["encoder_state_dict"]
        model_state = model.state_dict()
        filtered = {k: v for k, v in enc_state.items() if k in model_state}
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded encoder_state_dict into model. Copied params: {len(filtered)}")
        missing = [k for k in model_state.keys() if k not in filtered]
        if missing:
            print(f"  Remaining params will use initialization: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(
            f"Loaded raw state_dict. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")


def build_val_dataloader(config, batch_size=4, num_workers=0):
    input_dir = config.get("DATA", {}).get("input_dir", "data/jta_3dp_row")
    data_dir = os.path.join(input_dir, "val")
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def _to_cpu_mask_indices(mask_indices):
    cpu_indices = []
    for indices in mask_indices:
        if torch.is_tensor(indices):
            cpu_indices.append(indices.detach().cpu())
        else:
            cpu_indices.append(torch.tensor(indices))
    return cpu_indices


def evaluate_coordinate_reconstruction(
    model,
    dataloader,
    device,
    mask_ratio=0.3,
    max_batches=5,
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

            reconstructed = model(masked_batch)

            loss, loss_dict = compute_reconstruction_loss(
                batch,
                reconstructed,
                mask_indices,
                loss_type=loss_type,
                beta=beta,
            )

            total_losses.append(loss.item())
            masked_losses.append(loss_dict["masked_joints_loss_mean"])
            per_batch_stats.append(loss_dict)

            originals.append(batch.detach().cpu())
            reconstructions.append(reconstructed.detach().cpu())
            all_mask_indices.extend(_to_cpu_mask_indices(mask_indices))

            if save_dir is not None:
                batch_dir = os.path.join(save_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)

                plot_reconstruction_comparison(
                    batch.detach().cpu(),
                    masked_batch.detach().cpu(),
                    reconstructed.detach().cpu(),
                    mask_indices,
                    os.path.join(batch_dir, "comparison.png"),
                    overlay=False,
                )

                plot_reconstruction_comparison(
                    batch.detach().cpu(),
                    masked_batch.detach().cpu(),
                    reconstructed.detach().cpu(),
                    mask_indices,
                    os.path.join(batch_dir, "comparison_overlay.png"),
                    overlay=True,
                )

                plot_sequence_reconstruction_comparison(
                    batch.detach().cpu(),
                    masked_batch.detach().cpu(),
                    reconstructed.detach().cpu(),
                    mask_indices,
                    os.path.join(batch_dir, "sequence.png"),
                    overlay=False,
                )

                plot_sequence_reconstruction_comparison(
                    batch.detach().cpu(),
                    masked_batch.detach().cpu(),
                    reconstructed.detach().cpu(),
                    mask_indices,
                    os.path.join(batch_dir, "sequence_overlay.png"),
                    overlay=True,
                )

            if (batch_idx + 1) % log_every == 0:
                print(
                    f"[Batch {batch_idx + 1}] "
                    f"loss={loss.item():.6f}, masked={loss_dict['masked_joints_loss_mean']:.6f}"
                )

    if len(originals) == 0:
        raise RuntimeError("No batches evaluated.")

    original_cat = torch.cat(originals, dim=0)
    recon_cat = torch.cat(reconstructions, dim=0)

    stats = calculate_masked_unmasked_batch_errors(
        original_cat, recon_cat, all_mask_indices
    )

    summary = {
        "per_batch": per_batch_stats,
        **stats,
    }

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--cfg", type=str, default="configs_skeleton.yml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1", "rce"])
    parser.add_argument("--beta", type=float, default=2.0, help="RCE beta (used when loss_type='rce')")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="skeleton/val_coord_visualizations",
        help="Directory to save visualization images",
    )
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(args.cfg):
        with open(args.cfg, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    device = torch.device(args.device)

    feature_dim = config.get("MODEL", {}).get("feature_dim", 256)
    model = STGCN18Reconstructor(
        in_channels=3,
        out_channels=3,
        feature_dim=feature_dim,
    ).to(device)

    load_checkpoint_flex(model, args.ckpt, map_location=device)

    print("################",model.mask_token,"################")

    dataloader = build_val_dataloader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nSaving visualizations to: {args.save_dir}")
    stats = evaluate_coordinate_reconstruction(
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

    print("\n=== Coordinate Reconstruction on VAL ===")
    print(f"Masked micro mean:      {stats['micro_masked_mean']:.6f} m")
    print(f"Unmasked micro mean:    {stats['micro_unmasked_mean']:.6f} m")
    print(f"Masked macro mean:      {stats['macro_masked_mean']:.6f} m")
    print(f"Unmasked macro mean:    {stats['macro_unmasked_mean']:.6f} m")
    print(f"Masked instances:       {stats['total_masked_instances']}")
    print(f"Unmasked instances:     {stats['total_unmasked_instances']}")
    print(f"Avg masked/sample:      {stats['avg_masked_per_sample']:.2f} ({stats['mask_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()

