"""Train the baseline PyTorch UltraNeRF ultrasound renderer.

This is the main training entry point for the repository. It loads tracked
ultrasound frames and poses, constructs the NeRF-style PyTorch model via
``create_nerf()``, renders full ultrasound images from sampled 3D points, and
optimizes image-space losses against the target frames.

Outputs:
- checkpoints under ``logs/<expname>/``
- optional TensorBoard summaries
- periodic rendered training visualizations
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if ROOT.name == "scripts":
    SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import inspect
import os
import time
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ultranerf.load_us import load_us_data
from ultranerf.nerf_utils import create_nerf, img2mse, render_us, compute_loss, compute_regularization
from ultranerf.probe_geometry import build_probe_geometry_from_args, remap_image_to_convex_grid
from ultranerf.training_config import (
    DatasetSplit,
    apply_training_scheme_overrides,
    resolve_dataset_split,
    resolve_training_scheme,
    write_resolved_training_metadata,
)
from ultranerf.unerf_config import config_parser
from ultranerf.visualization.comparison_panel import normalize_recorded_image_for_display
from ultranerf.visualization.render_panel import normalize_image_for_display

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_PREVIEW_SEPARATOR_PX = 24
VALIDATION_PREVIEW_SAMPLE_COUNT = 6
VALIDATION_PREVIEW_FRAME_DURATION_MS = 350


def _select_training_and_validation_indices(
    args,
    *,
    frame_count: int,
    default_holdout: int | list[int],
) -> tuple[np.ndarray, np.ndarray, DatasetSplit | None]:
    split = resolve_dataset_split(getattr(args, "split_file", None), frame_count)
    if split is not None:
        return (
            np.array(split.train_indices, dtype=np.int64),
            np.array(split.validation_indices, dtype=np.int64),
            split,
        )

    if not isinstance(default_holdout, list):
        default_holdout = [default_holdout]
    validation = np.array(default_holdout, dtype=np.int64)
    training = np.array(
        [index for index in np.arange(int(frame_count)) if index not in set(validation.tolist())],
        dtype=np.int64,
    )
    return training, validation, None


def _append_progress_event(run_dir: str | Path, payload: dict[str, object]) -> Path:
    progress_path = Path(run_dir) / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return progress_path


def _compose_validation_preview_frame(
    *,
    target: torch.Tensor,
    output_image: torch.Tensor,
    separator_px: int = VALIDATION_PREVIEW_SEPARATOR_PX,
) -> np.ndarray:
    target_buffer = normalize_recorded_image_for_display(target.detach().cpu().numpy()[0, 0])
    render_buffer = normalize_image_for_display(output_image.detach().cpu().numpy()[0, 0])
    if target_buffer.shape != render_buffer.shape:
        raise ValueError(
            f"Validation preview expects matching target/render shapes, got {target_buffer.shape} and {render_buffer.shape}"
        )
    separator = np.zeros((target_buffer.shape[0], int(separator_px)), dtype=np.uint8)
    return np.concatenate([target_buffer, separator, render_buffer], axis=1)


def _sample_validation_preview_indices(
    validation_indices: np.ndarray,
    *,
    preferred_index: int,
    sample_count: int = VALIDATION_PREVIEW_SAMPLE_COUNT,
) -> list[int]:
    if len(validation_indices) == 0:
        return []
    max_count = max(1, min(int(sample_count), len(validation_indices)))
    positions = np.linspace(0, len(validation_indices) - 1, num=max_count, dtype=np.int64)
    selected = [int(validation_indices[pos]) for pos in positions]
    preferred = int(preferred_index)
    if preferred in validation_indices.tolist() and preferred not in selected:
        selected[0] = preferred
    deduped: list[int] = []
    for idx in selected:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def _save_validation_preview(
    *,
    basedir: str,
    expname: str,
    step: int,
    frames: list[np.ndarray],
) -> Path:
    output_dir = Path(basedir) / expname / "validation_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Validation preview requires at least one frame")
    pil_frames = [Image.fromarray(frame, mode="L") for frame in frames]
    step_path = output_dir / f"{step:08d}.gif"
    latest_path = output_dir / "latest.gif"
    pil_frames[0].save(
        step_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=VALIDATION_PREVIEW_FRAME_DURATION_MS,
        loop=0,
    )
    pil_frames[0].save(
        latest_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=VALIDATION_PREVIEW_FRAME_DURATION_MS,
        loop=0,
    )
    return latest_path


def train():

    parser = config_parser()
    args = parser.parse_args()
    training_scheme = resolve_training_scheme(getattr(args, "training_scheme", None))
    args = apply_training_scheme_overrides(args, training_scheme)
    probe_geometry = build_probe_geometry_from_args(args)
    if probe_geometry.is_convex:
        args.N_samples = int(probe_geometry.convex_n_samples)

    if args.random_seed == 0:
        print("Setting deterministic behaviour")
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if args.dataset_type == "us":
        # IT CONVERTS THE POSE TRANSLATION FROM MM TO M ALREADY!!!
        images, poses, i_test = load_us_data(
            args.datadir, confmap=args.confmap, pose_path=args.pose_path
        )
        all_indices = np.arange(images.shape[0])
        i_test = all_indices[::10].tolist()  # [0, 10, 20, 30, ...]

        i_train, i_val, explicit_split = _select_training_and_validation_indices(
            args,
            frame_count=int(images.shape[0]),
            default_holdout=i_test,
        )
        i_test = list(i_val.tolist())
        print("Validation {}, train {}".format(len(i_val), len(i_train)))

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    raw_H, raw_W = images.shape[1], images.shape[2]
    if probe_geometry.is_convex:
        H = int(probe_geometry.convex_n_samples)
        W = int(probe_geometry.convex_n_rays)
        sy = float(probe_geometry.convex_scale_y_mm) * scaling
        sx = float(probe_geometry.convex_scale_x_mm) * scaling
        far = (probe_geometry.convex_outer_radius_mm - probe_geometry.convex_inner_radius_mm) * scaling
    else:
        probe_depth = args.probe_depth * scaling
        probe_width = args.probe_width * scaling
        far = probe_depth
        H, W = raw_H, raw_W
        sy = probe_depth / float(H)
        sx = probe_width / float(W)
    sh = sy
    sw = sx

    basedir = args.basedir
    expname = args.expname

    # Create tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname))

    # Create log dir and copy the config file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())
    resolved_training_path = write_resolved_training_metadata(
        Path(basedir) / expname,
        args=args,
        training_scheme=training_scheme,
        dataset_split=explicit_split,
    )

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, optimizer, _ = create_nerf(
        args, device=device
    )

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_iters = args.n_iters
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    N_splits = 20  # Training を20分割
    steps_per_split = max(1, N_iters // N_splits) 
    shuffled_i_train = np.random.permutation(i_train) 

    val_loss_history = []
    pool_percentages_history = []

    _append_progress_event(
        Path(basedir) / expname,
        {
            "event": "started",
            "step": int(start),
            "total_steps": int(N_iters),
            "training_scheme": None if training_scheme is None else training_scheme.name,
            "resolved_training_config_path": str(resolved_training_path.resolve()),
            "train_count": int(len(i_train)),
            "validation_count": int(len(i_val)),
        },
    )

    # Losses
    ssim_weight = args.ssim_lambda
    l2_weight = 1.0 - ssim_weight
    ssim_kwargs = {
        "spatial_dims": 2,
        "win_size": args.ssim_filter_size,
        "k1": 0.01,
        "k2": 0.1,
    }
    ssim_signature = inspect.signature(SSIMLoss)
    if "data_range" in ssim_signature.parameters:
        ssim_kwargs["data_range"] = 1.0
    if "kernel_type" in ssim_signature.parameters:
        ssim_kwargs["kernel_type"] = "gaussian"
    ssim_loss = SSIMLoss(**ssim_kwargs)
    losses = {"l2": img2mse,
              "ssim": ssim_loss,
              "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2)}
    start = start + 1
    # render_kwargs_train["pts"] = None
    for i in trange(start, N_iters + 1):
        time0 = time.time()

        current_split_idx = min(N_splits - 1, (i - 1) // steps_per_split)
        current_percentage = (current_split_idx + 1) / N_splits
        current_pool_size = max(1, int(len(i_train) * current_percentage))
        
        # 現在のフェーズでアクセス可能な画像のサブセット
        active_train_pool = shuffled_i_train[:current_pool_size]

        img_i = np.random.choice(
           active_train_pool 
        )  # Why? This does not guarantee that all images are used --> probably a weighted random would be better,
        # or removing from a temporary set as long as it's not empty

        target_array = images[img_i]
        if probe_geometry.is_convex:
            target_array = remap_image_to_convex_grid(target_array, probe_geometry)
        target = torch.Tensor(target_array).to(device).unsqueeze(0).unsqueeze(0)
        pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)

        #####  Core optimization loop  #####
        rendering_output = render_us(
            H, W, sw, sh, c2w=pose, chunk=args.chunk, retraw=True, **render_kwargs_train
        )
        output_image = rendering_output["intensity_map"]

        optimizer.zero_grad()

        loss = compute_loss(output_image, target, args, losses, i, training_scheme=training_scheme)
        if args.reg and i > args.r_warm_up_it:
            reg = compute_regularization(rendering_output, losses,
                                      weights=(args.r_lcc_penalty, args.r_tv_penalty, args.r_max_reflection))
            loss = {**loss, **reg}

        total_loss = 0.0
        for loss_value in loss.values():
            tmp = loss_value[0] * loss_value[1]
            total_loss += tmp

        if type(total_loss) != torch.Tensor:
            raise ValueError("Loss is not a tensor: Problem with loss calculation")

        total_loss.backward()
        optimizer.step()

        dt = time.time() - time0

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        if args.tensorboard:
            writer.add_scalar("Loss/total_loss", total_loss.item(), i)
            for k, v in loss.items():
                writer.add_scalar(f"Loss/{k}", v[1].item(), i)
            writer.add_scalar("Learning rate", new_lrate, i)

        if len(i_val) > 0 and args.validation_preview_every > 0 and (i + 1) % int(args.validation_preview_every) == 0:
            val_idx = int(i_val[int(args.validation_preview_index) % len(i_val)])
            sampled_validation_indices = _sample_validation_preview_indices(
                i_val,
                preferred_index=val_idx,
            )
            preview_frames: list[np.ndarray] = []
            with torch.no_grad():
                for current_val_idx in sampled_validation_indices:
                    val_target_array = images[current_val_idx]
                    if probe_geometry.is_convex:
                        val_target_array = remap_image_to_convex_grid(val_target_array, probe_geometry)
                    val_target = torch.Tensor(val_target_array).to(device).unsqueeze(0).unsqueeze(0)
                    val_pose = torch.from_numpy(poses[current_val_idx, :3, :4]).to(device).unsqueeze(0)
                    validation_output = render_us(
                        H, W, sw, sh, c2w=val_pose, chunk=args.chunk, retraw=True, **render_kwargs_test
                    )
                    preview_frames.append(
                        _compose_validation_preview_frame(
                            target=val_target,
                            output_image=validation_output["intensity_map"],
                        )
                    )
            preview_path = _save_validation_preview(
                basedir=basedir,
                expname=expname,
                step=i + 1,
                frames=preview_frames,
            )
            _append_progress_event(
                Path(basedir) / expname,
                {
                    "event": "validation_preview",
                    "step": int(i + 1),
                    "total_steps": int(N_iters),
                    "validation_index": val_idx,
                    "validation_indices": list(sampled_validation_indices),
                    "preview_path": str(preview_path.resolve()),
                    "training_loss": float(total_loss.item()),
                },
            )

        dt = time.time() - time0
        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(
                os.path.join(basedir, expname, "train_rendering"), exist_ok=True
            )

            print(f"Step: {i+1}, Loss: {total_loss.item()}, Time: {dt}")  # type: ignore
            detailed_loss_string = ", ".join(
                [f"{k}: {v[1].item()}" for k, v in loss.items()]
            )
            print(detailed_loss_string)

            plt.figure(figsize=(16, 8))
            for j, m in enumerate(rendering_output):

                plt.subplot(3, 4, j + 1)
                plt.title(m)
                plt.imshow(rendering_output[m].detach().cpu().numpy()[0, 0].T)

            plt.subplot(3, 4, 12)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy()[0, 0].T)

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()
            _append_progress_event(
                Path(basedir) / expname,
                {
                    "event": "progress",
                    "step": int(i + 1),
                    "total_steps": int(N_iters),
                    "training_loss": float(total_loss.item()),
                    "learning_rate": float(new_lrate),
                    "loss_terms": {k: float(v[1].item()) for k, v in loss.items()},
                },
            )

        if (i + 1) % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i + 1))
            torch.save(
                {
                    "global_step": i,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            _append_progress_event(
                Path(basedir) / expname,
                {
                    "event": "checkpoint",
                    "step": int(i + 1),
                    "total_steps": int(N_iters),
                    "checkpoint_path": str(Path(path).resolve()),
                },
            )

        if i % steps_per_split == 0 or i == N_iters:
            print(f"\n[Phase {current_split_idx + 1}/{N_splits}] Evaluating Val Loss (Pool: {current_percentage*100:.1f}%, {current_pool_size}/{len(i_train)} imgs)...")
            val_losses = []
            with torch.no_grad():
                for v_idx in i_val:
                    v_target_arr = images[v_idx]
                    if probe_geometry.is_convex:
                        v_target_arr = remap_image_to_convex_grid(v_target_arr, probe_geometry)
                    v_target = torch.Tensor(v_target_arr).to(device).unsqueeze(0).unsqueeze(0)
                    v_pose = torch.from_numpy(poses[v_idx, :3, :4]).to(device).unsqueeze(0)
                    
                    v_out = render_us(H, W, sw, sh, c2w=v_pose, chunk=args.chunk, retraw=True, **render_kwargs_test)
                    v_loss_dict = compute_loss(v_out["intensity_map"], v_target, args, losses, i, training_scheme=training_scheme)
                    v_tot = sum(lv[0] * lv[1] for lv in v_loss_dict.values())
                    val_losses.append(v_tot.item())
            
            mean_v_loss = np.mean(val_losses)
            val_loss_history.append(mean_v_loss)
            pool_percentages_history.append(current_percentage * 100)
            
            print(f"--> Mean Val Loss: {mean_v_loss:.6f}")
            if args.tensorboard:
                writer.add_scalar("Experiment/Validation_Loss", mean_v_loss, i)

    _append_progress_event(
        Path(basedir) / expname,
        {
            "event": "completed",
            "step": int(N_iters),
            "total_steps": int(N_iters),
        },
    )

    plt.figure(figsize=(8, 5))
    plt.plot(pool_percentages_history, val_loss_history, marker='o', linestyle='-', color='b')
    plt.title("Validation Loss vs Active Training Data Pool")
    plt.xlabel("Percentage of Data Used (%)")
    plt.ylabel("Mean Validation Loss")
    plt.grid(True)
    plot_path = os.path.join(basedir, expname, "data_efficiency_curve.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close()
    
    with open(os.path.join(basedir, expname, "data_efficiency_results.json"), "w") as f:
        json.dump({"percentages": pool_percentages_history, "val_losses": val_loss_history}, f, indent=4)
    print(f"\n[Success] Data efficiency plot saved to {plot_path}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_default_device") and torch.cuda.is_available():
        torch.set_default_device("cuda")
    train()
