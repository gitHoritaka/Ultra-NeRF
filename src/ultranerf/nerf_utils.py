import os

import numpy as np
import torch
import torch.nn as nn

from ultranerf.convex_mip import mip_input_channels, render_rays_us_convex_mip
from ultranerf.model import NeRF, BARF, PoseRefine, Reconstruction
from ultranerf.probe_geometry import ProbeGeometry, build_probe_geometry_from_args
from ultranerf.rendering import render_rays_us, render_rays_us_with_reconstruction, render_rays_us_with_reconstruction_pts

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]
        B = self.kwargs["B"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(
                0.0, max_freq, steps=N_freqs, device=self.kwargs["device"]
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**max_freq, steps=N_freqs, device=self.kwargs["device"]
            )

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                if B is not None:
                    embed_fns.append(
                        lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(
                            x @ torch.transpose(B, 0, 1) * freq
                        )
                    )
                    out_dim += d
                    out_dim += B.shape[1]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, device, i=0, b=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    if b != 0:
        B = torch.randn(size=(b, 3))
    else:
        B = None

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dim,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
        "B": B,
        "device": device,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = torch.arange(-W / 2, W / 2, dtype=torch.float32, device=c2w.device) * sw
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    origin_base = torch.stack([x, y, z], dim=1).to(c2w.device)

    origin_rotated = R @ origin_base.transpose(
        0, 1
    )  # THIS WAS HADAMARD PRODUCT IN THE ORIGINAL CODE !!!
    ray_o_r = origin_rotated.transpose(0, 1)
    rays_o = ray_o_r + t

    dirs_base = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=c2w.device)
    dirs_r = R @ dirs_base
    rays_d = dirs_r.expand_as(rays_o)

    return rays_o, rays_d


def get_rays_us_convex(
    geometry: ProbeGeometry,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate convex fan rays in world coordinates."""
    if not geometry.is_convex:
        raise ValueError("get_rays_us_convex requires a convex probe geometry")
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    fan_half = torch.deg2rad(torch.tensor(float(geometry.convex_angle_deg), dtype=torch.float32, device=c2w.device)) * 0.5
    angles = torch.linspace(-fan_half, fan_half, steps=int(geometry.convex_n_rays), device=c2w.device)
    dirs_base = torch.stack(
        [torch.sin(angles), torch.cos(angles), torch.zeros_like(angles)],
        dim=-1,
    )
    origins_base = dirs_base.clone()
    origins_base[:, 0] *= float(geometry.convex_inner_radius_mm) * 0.001
    origins_base[:, 1] *= float(geometry.convex_inner_radius_mm) * 0.001

    rays_o = (R @ origins_base.transpose(0, 1)).transpose(0, 1) + t
    rays_d = (R @ dirs_base.transpose(0, 1)).transpose(0, 1)
    return rays_o, rays_d


def get_rays_us(
    H: int,
    W: int,
    sw: float,
    sh: float,
    c2w: torch.Tensor,
    probe_geometry: ProbeGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch ray generation by probe geometry."""
    geometry = probe_geometry or ProbeGeometry(width_mm=float(W) * float(sw) * 1000.0, depth_mm=float(H) * float(sh) * 1000.0)
    if geometry.is_convex:
        return get_rays_us_convex(geometry, c2w)
    return get_rays_us_linear(H, W, sw, sh, c2w)


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # Gather the neighboring CDF and bin values for inverse transform sampling.
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, fn, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def run_network_preencoded(inputs, fn, netchunk=1024 * 64):
    """Apply a network to pre-encoded features without positional embedding."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    outputs_flat = batchify(fn, netchunk)(inputs_flat)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def run_barf_network(inputs, fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    outputs_flat = batchify(fn, netchunk)(inputs_flat)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    render_mode = kwargs.get("render_mode", "default")
    if kwargs["network_rec"] is not None:
        renderer = render_rays_us_with_reconstruction
    elif render_mode == "convex_mip":
        renderer = render_rays_us_convex_mip
        kwargs = dict(kwargs)
        kwargs["multires"] = int(kwargs.pop("mip_multires"))
        kwargs["use_elongation"] = bool(kwargs.pop("mip_use_elongation"))
        kwargs["max_elongation"] = float(kwargs.pop("mip_max_elongation"))
        kwargs["pixel_radius"] = float(kwargs.pop("mip_pixel_radius"))
    else:
        renderer = render_rays_us
    try:
        if kwargs["pts"] is not None:
            renderer = render_rays_us_with_reconstruction_pts
            rays_flat = kwargs['pts']
    except KeyError:
        pass
    for i in range(0, rays_flat.shape[0], chunk):
        ret = renderer(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def create_nets_for_reconstruction(args, device, mode="train"):
    """Instantiate NeRF's MLP model."""
    if getattr(args, "render_mode", "default") == "convex_mip":
        raise NotImplementedError("Reconstruction mode is not supported for render_mode=convex_mip")
    probe_geometry = build_probe_geometry_from_args(args)
    if probe_geometry.is_convex:
        args.N_samples = int(probe_geometry.convex_n_samples)
    embed_fn, input_ch = get_embedder(args.multires, device, args.i_embed)
    if args.rec_only_theta or args.rec_only_occ:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=3)
    else:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=6)
    output_ch = args.output_ch
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
    ).to(device)

    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, network_fn: run_network(
        inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
    )

    network_query_fn_rec = lambda inputs, network_fn: run_network(
        inputs, network_fn, embed_fn=embed_fn_rec, netchunk=args.netchunk
    )
    if args.reconstruction:
        model_rec = Reconstruction(
            D=args.netdepth,
            W=args.netwidth,
            input_ch= input_ch_rec,
            output_ch=1,
            skips=skips,
        ).to(device)

        # for i, l in enumerate(model.pts_linears):
        #     if i < 6:
        #         for param in l.parameters():
        #             param.requires_grad = False

        grad_vars_reg = list(model_rec.parameters())
        optimizer_reg = torch.optim.Adam(params=grad_vars_reg, lr=args.lrate, betas=(0.9, 0.999))
    else:
        model_rec = None
        optimizer_reg = None
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer_reg.load_state_dict(ckpt["optimizer_rec_state_dict"])

        # remove paramerts "views_linears.0.weight", "views_linears.0.bias" from state_dict
        # if exists, this is for compatibility with the old code

        new_state_dict_rec = {}
        for k, v in ckpt["network_rec_state_dict"].items():
            if "views_linears.0" not in k:
                new_state_dict_rec[k] = v
        model_rec.load_state_dict(new_state_dict_rec)
    ckpts_nerf = [
        os.path.join(basedir, args.expname_nerf, f)
        for f in sorted(os.listdir(os.path.join(basedir, args.expname_nerf)))
        if "tar" in f
    ]
    nerf_weights = torch.load(ckpts_nerf[-1])
    # Load NeRF model
    new_state_dict = {}
    for k, v in nerf_weights["network_fn_state_dict"].items():
        if "views_linears.0" not in k:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "network_query_fn_rec": network_query_fn_rec,
        "N_samples": args.N_samples,
        "network_fn": model,
        "network_rec": model_rec,
        "probe_geometry": probe_geometry,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_reg




def create_nerf(args, device, mode="train"):
    """Instantiate NeRF's MLP model."""
    probe_geometry = build_probe_geometry_from_args(args)
    render_mode = getattr(args, "render_mode", "default")
    if render_mode == "convex_mip" and not probe_geometry.is_convex:
        raise ValueError("render_mode=convex_mip requires probe_type=convex")
    if probe_geometry.is_convex:
        args.N_samples = int(probe_geometry.convex_n_samples)
    embed_fn, input_ch = get_embedder(args.multires, device, args.i_embed)
    if render_mode == "convex_mip":
        input_ch = mip_input_channels(args.multires)
    if args.rec_only_theta or args.rec_only_occ:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=3)
    else:
        embed_fn_rec, input_ch_rec = get_embedder(args.multires, device, args.i_embed, input_dim=6)
    output_ch = args.output_ch
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
    ).to(device)

    grad_vars = list(model.parameters())

    if render_mode == "convex_mip":
        network_query_fn = lambda inputs, network_fn: run_network_preencoded(
            inputs, network_fn, netchunk=args.netchunk
        )
    else:
        network_query_fn = lambda inputs, network_fn: run_network(
            inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
        )

    network_query_fn_rec = lambda inputs, network_fn: run_network(
        inputs, network_fn, embed_fn=embed_fn_rec, netchunk=args.netchunk
    )
    if args.reconstruction:
        model_rec = Reconstruction(
            D=args.netdepth,
            W=args.netwidth,
            input_ch= input_ch_rec,
            output_ch=1,
            skips=skips,
        ).to(device)

        grad_vars_reg = list(model_rec.parameters())
        optimizer_reg = torch.optim.Adam(params=grad_vars_reg, lr=args.lrate, betas=(0.9, 0.999))
    else:
        model_rec = None
        optimizer_reg = None
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]

        if mode == "train":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if args.reconstruction:
                optimizer_reg.load_state_dict(ckpt["optimizer_rec_state_dict"])

        # remove paramerts "views_linears.0.weight", "views_linears.0.bias" from state_dict
        # if exists, this is for compatibility with the old code
        new_state_dict = {}
        for k, v in ckpt["network_fn_state_dict"].items():
            if "views_linears.0" not in k:
                new_state_dict[k] = v
        if args.reconstruction:
            new_state_dict_rec = {}
            for k, v in ckpt["network_rec_state_dict"].items():
                if "views_linears.0" not in k:
                    new_state_dict_rec[k] = v
            model_rec.load_state_dict(new_state_dict_rec)

        # Load model
        model.load_state_dict(new_state_dict)


    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "network_query_fn_rec": network_query_fn_rec,
        "N_samples": args.N_samples,
        "network_fn": model,
        "network_rec": model_rec,
        "probe_geometry": probe_geometry,
        "render_mode": render_mode,
        "mip_multires": int(args.multires),
        "mip_use_elongation": bool(getattr(args, "mip_use_elongation", False)),
        "mip_max_elongation": float(getattr(args, "mip_max_elongation", 5.0)),
        "mip_pixel_radius": float(getattr(args, "mip_pixel_radius", 0.00111)),
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer, optimizer_reg


def create_barf(poses: torch.Tensor, args, device, mode="train"):
    """Instantiate BARF's MLP model."""

    output_ch = args.output_ch
    skips = [4]
    input_ch = 3

    model = BARF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        L=args.L,
    ).to(device)

    pose_refine = PoseRefine(poses=poses, mode=mode).to(device)

    network_query_fn = lambda inputs, network_fn: run_barf_network(
        inputs, network_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999)
    )

    pose_optim = torch.optim.Adam(pose_refine.parameters(), args.pose_lr)
    gamma = (args.pose_lr_end / args.pose_lr) ** (1.0 / args.n_iters)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_optim, gamma)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        pose_optim.load_state_dict(ckpt["pose_optim_state_dict"])
        pose_sched.load_state_dict(ckpt["pose_sched_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        pose_refine.load_state_dict(ckpt["pose_refine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "N_samples": args.N_samples,
        "network_fn": model,
        "pose_refine": pose_refine,
        "ckpt": ckpt if len(ckpts) > 0 else None,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return (
        render_kwargs_train,
        render_kwargs_test,
        start,
        optimizer,
        pose_optim,
        pose_sched,
    )


def render_us(
    H,
    W,
    sw,
    sh,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=55.0 * 0.001,
    **kwargs
):
    """Render rays."""
    try:
        if kwargs["pts"] is not None:
            all_ret = batchify_rays(rays, chunk=chunk, **kwargs)
            # for k in all_ret:
            #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            #     all_ret[k] = all_ret[k].reshape(k_sh)
            return all_ret
    except KeyError:
        pass
    # assert rays is not None or c2w is not None
    if rays is None and c2w is None:
        raise ValueError("rays and c2w are both None")

    rays_o = None
    rays_d = None

    probe_geometry = kwargs.pop("probe_geometry", None)
    if c2w is not None:
        # Special case to render full image
        for c in c2w:
            if rays_o is None and rays_d is None:
                rays_o, rays_d = get_rays_us(H, W, sw, sh, c, probe_geometry=probe_geometry)
            else:
                o, d = get_rays_us(H, W, sw, sh, c, probe_geometry=probe_geometry)
                rays_o = torch.concatenate((rays_o, o))  # type: ignore
                rays_d = torch.concatenate((rays_d, d))  # type: ignore
    else:
        # Use provided ray batch
        rays_o, rays_d = rays

    if rays_o is None:
        raise ValueError("rays_o is None")
    if rays_d is None:
        raise ValueError("rays_d is None")

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray

    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk=chunk, **kwargs)
    # for k in all_ret:
    #     k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
    #     all_ret[k] = all_ret[k].reshape(k_sh)
    return all_ret


def compute_loss(output, target, args, losses, i, training_scheme=None):
    loss = {}

    if training_scheme is not None:
        active_terms = training_scheme.active_loss_terms(int(i))
        if not active_terms:
            raise ValueError(f"Training scheme {training_scheme.name} produced no active losses at step {i}")
        key_counts = {}
        for term in active_terms:
            if term.name not in losses:
                raise KeyError(f"Loss '{term.name}' is not available in the training runtime")
            base_key = term.resolved_key
            suffix = key_counts.get(base_key, 0)
            key_counts[base_key] = suffix + 1
            key = base_key if suffix == 0 else f"{base_key}_{suffix + 1}"
            loss[key] = (float(term.weight), losses[term.name](output, target))
        return loss

    if args.loss == "l2" or i < args.r_warm_up_it:
        l2_intensity_loss = losses['l2'](output, target)
        loss["l2"] = (1.0, l2_intensity_loss)
    elif args.loss == "ssim":
        ssim_intensity_loss = losses['ssim'](output, target)
        loss["ssim"] = (args.ssim_lambda, ssim_intensity_loss)
        l2_intensity_loss = img2mse(output, target)
        loss["l2"] = (1.-args.ssim_lambda, l2_intensity_loss)

    return loss

def compute_regularization(rendering_output, reg_funcs, weights=(0.01, 0.00001, 0.34)):
    lncc = reg_funcs['lncc']
    lncc_w, tv_w, refl_max = weights
    lcc_penalty_scatter_attenuation = lncc(
        rendering_output['scatter_amplitude'],
        rendering_output['attenuation_coeff'])
    reg = {}

    reg["lcc_penalty"] = (lncc_w, lcc_penalty_scatter_attenuation)
    dy_ampl = rendering_output['scatter_amplitude'][:, :, :, 1:] - rendering_output['scatter_amplitude'][:, :, :, :-1]
    dy_ampl = torch.cat([dy_ampl, dy_ampl[:, :, :, -1:]], dim=-1)  # Pad y direction

    dx_ampl = rendering_output['scatter_amplitude'][:, :, 1:, :] - rendering_output['scatter_amplitude'][:, :, :-1, :]
    dx_ampl = torch.cat([dx_ampl, dx_ampl[:, :, -1:, :]], dim=-2)
    # Calculate TV penalties
    total_variation_penalty_y_ampl = torch.sum(
        (refl_max - rendering_output['reflection_coeff'])
        * torch.abs(dy_ampl.squeeze()))
    total_variation_penalty_x_ampl = torch.sum(
        (refl_max - rendering_output['reflection_coeff'])
        * torch.abs(dx_ampl.squeeze()))

    amplitude_tv_penalty = total_variation_penalty_x_ampl + total_variation_penalty_y_ampl
    reg["tv_penalty"] = (tv_w, amplitude_tv_penalty)
    return reg

def compute_pts_from_pose(H, W, sw, sh, pose, near, far):
    o, d = get_rays_us(H, W, sw, sh, pose)
    o = o.reshape(-1, 3).float()
    d = d.reshape(-1, 3).float()
    # Decide where to sample along each ray
    N_samples = H
    t_vals = torch.linspace(0.0, 1.0, N_samples).to(pose.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals

    z_vals = z_vals.expand(W, N_samples)

    # Points in space to evaluate model at
    origin = o.unsqueeze(-2)
    step = d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    pts = step + origin

    return pts
