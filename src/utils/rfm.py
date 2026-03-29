import torch


def prepare_rfm_batch(
    x1: torch.Tensor, mask_bool: torch.Tensor, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples time t, generates noise x0, and computes the intermediate state xt
    and target velocity for Rectified Flow Matching.
    """
    batch_size = x1.shape[0]
    t = torch.rand((batch_size,), device=device)
    t_expanded = t.view(-1, 1, 1)

    x0 = torch.randn_like(x1)
    xt_hole = t_expanded * x1 + (1.0 - t_expanded) * x0

    # Context condition
    xt = torch.where(mask_bool.expand_as(x1), xt_hole, x1)
    target_v = x1 - x0

    return xt, target_v, t


def sample_euler(
    model: torch.nn.Module,
    x1_context: torch.Tensor,
    mask_bool: torch.Tensor,
    text_emb: torch.Tensor,
    text_mask: torch.Tensor,
    mel_pad_mask: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """
    Solves the ODE using Euler's method to generate the inpainted spectrogram.
    Starts from t=0 (pure noise) and integrates to t=1 (clean audio).
    """
    device = x1_context.device
    batch_size = x1_context.shape[0]

    x_t = torch.randn_like(x1_context)
    noise_for_context = x_t.clone()

    dt = 1.0 / num_steps
    mask_float = mask_bool.to(torch.float32)
    uncond_text_emb = torch.zeros_like(text_emb)

    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.full((batch_size,), t_val, device=device)

        # Enforce context
        x_t_exact_context = t_val * x1_context + (1.0 - t_val) * noise_for_context
        x_t = torch.where(mask_bool, x_t, x_t_exact_context)

        with torch.no_grad():
            v_cond = model(xt=x_t, mask=mask_float, t=t, text_emb=text_emb, text_mask=text_mask,
                           mel_pad_mask=mel_pad_mask)

            if cfg_scale == 1.0:
                v_pred = v_cond
            else:
                v_uncond = model(xt=x_t, mask=mask_float, t=t, text_emb=uncond_text_emb, text_mask=text_mask,
                                 mel_pad_mask=mel_pad_mask)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        x_t = x_t + v_pred * dt

    x_t = torch.where(mask_bool, x_t, x1_context)
    x_t = torch.clamp(x_t, min=-1.0, max=1.0)
    return x_t
