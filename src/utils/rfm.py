import torch
from tqdm import tqdm


def prepare_rfm_batch(
    x1: torch.Tensor, mask_bool: torch.Tensor, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples time t, generates noise x0, and computes the intermediate state xt
    and target velocity for Rectified Flow Matching.

    :param x1: Normalized mel-spectrogram in [-1, 1] range. Shape: [Batch, Mel_Bins, Time].
    :param mask_bool: Boolean mask where True indicates the hole region. Shape: [Batch, 1, Time].
    :param device: Target device for tensor operations.
    :return: Tuple of (xt, x_context, target_v, t) where:
        - xt: Interpolated noisy mel at timestep t. Shape: [Batch, Mel_Bins, Time].
        - x_context: Normalized mel with hole filled with -1.0 (silence). Shape: [Batch, Mel_Bins, Time].
        - target_v: Target velocity (x1 - x0) in hole regions, zeros elsewhere. Shape: [Batch, Mel_Bins, Time].
        - t: Sampled timesteps in [0, 1]. Shape: [Batch].
    """
    batch_size = x1.shape[0]
    t = torch.rand((batch_size,), device=device)
    t_expanded = t.view(-1, 1, 1)

    x0 = torch.randn_like(x1)
    x_context = torch.where(mask_bool.expand_as(x1), torch.full_like(x1, -1.0), x1)

    # Context condition
    xt = t_expanded * x1 + (1.0 - t_expanded) * x0
    target_v_hole = x1 - x0
    target_v = torch.where(mask_bool.expand_as(x1), target_v_hole, torch.zeros_like(x1))

    return xt, x_context, target_v, t


def sample_euler(
    model: torch.nn.Module,
    x1_context: torch.Tensor,
    mask_bool: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    verbose: bool = True,
    **condition_kwargs
) -> torch.Tensor:
    """
    Solves the probability flow ODE using Euler's method to generate an inpainted spectrogram.

    :param model: The DiT velocity prediction model.
    :param x1_context: Normalized mel-spectrogram in [-1, 1] used for contextual conditioning.
    :param mask_bool: Boolean mask where True indicates the hole to be generated.
    :param num_steps: Number of Euler integration steps (default: 50).
    :param cfg_scale: Classifier-free guidance scale. 1.0 disables CFG.
    :param verbose: Whether to display a progress bar during sampling.
    :param condition_kwargs: Additional conditioning features (phoneme_ids, text_mask, mel_pad_mask, cfg_drop_mask).
    :return: Generated normalized mel-spectrogram in [-1, 1].
    """
    device = x1_context.device
    batch_size = x1_context.shape[0]

    x_t = torch.randn_like(x1_context)
    noise_for_context = x_t.clone()

    dt = 1.0 / num_steps
    mask_float = mask_bool.to(torch.float32)
    x_context = torch.where(mask_bool.expand_as(x1_context), torch.full_like(x1_context, -1.0), x1_context)
    batched_x_context = torch.cat([x_context, x_context], dim=0)
    batched_mask_float = torch.cat([mask_float, mask_float], dim=0)
    batched_condition_kwargs = {}

    if cfg_scale != 1.0:
        cfg_drop_mask_cond = torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=device)
        cfg_drop_mask_uncond = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
        batched_cfg_drop_mask = torch.cat([cfg_drop_mask_cond, cfg_drop_mask_uncond], dim=0)

        for k, v in condition_kwargs.items():
            if isinstance(v, torch.Tensor):
                batched_condition_kwargs[k] = torch.cat([v, v], dim=0)
            else:
                batched_condition_kwargs[k] = v

        batched_condition_kwargs["cfg_drop_mask"] = batched_cfg_drop_mask
    else:
        condition_kwargs = {
            **condition_kwargs,
            "cfg_drop_mask": torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=device)
        }

    pbar = tqdm(range(num_steps), desc="Sampling") if verbose else range(num_steps)
    for i in pbar:
        t_val = i / num_steps
        t = torch.full((batch_size,), t_val, device=device)

        # Enforce context
        x_t_exact_context = t_val * x_context + (1.0 - t_val) * noise_for_context
        x_t = torch.where(mask_bool, x_t, x_t_exact_context)

        if cfg_scale == 1.0:
            with torch.no_grad():
                v_pred = model(xt=x_t, x_context=x_context, mask=mask_float, t=t, **condition_kwargs)
        else:
            batched_x_t = torch.cat([x_t, x_t], dim=0)
            batched_t = torch.cat([t, t], dim=0)

            with torch.no_grad():
                batched_v = model(
                    xt=batched_x_t,
                    x_context=batched_x_context,
                    mask=batched_mask_float,
                    t=batched_t,
                    **batched_condition_kwargs
                )

            v_cond, v_uncond = batched_v.chunk(2, dim=0)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        x_t = x_t + v_pred * dt

    x_t = torch.where(mask_bool, x_t, x1_context)
    x_t = torch.clamp(x_t, min=-1.0, max=1.0)
    return x_t
