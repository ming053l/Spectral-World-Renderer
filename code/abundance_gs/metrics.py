"""The canonical HSI metric of the paper (unified_hsi_metrics).

PSNR over all pixels (peak 1, prediction and reference clamped to [0, 1]); SAM over pixels whose reference norm
exceeds min_norm; plus foreground / background, correlation and derivative diagnostics.
"""
import math

import torch


def psnr_from_mse(mse: torch.Tensor) -> float:
    return float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())


def unified_hsi_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_norm: float,
) -> dict[str, float]:
    """Metrics with a support mask determined only by ground truth."""

    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")

    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    pred_norm = flat_pred.norm(dim=0)
    target_norm = flat_target.norm(dim=0)
    valid = target_norm > float(min_norm)
    background = ~valid
    if not bool(valid.any()):
        raise RuntimeError("The GT-defined support is empty.")

    full_error = pred - target
    full_mse = torch.mean(full_error.square())
    valid_error = flat_pred[:, valid] - flat_target[:, valid]
    valid_mse = torch.mean(valid_error.square())
    pred_valid = flat_pred[:, valid]
    target_valid = flat_target[:, valid]
    pred_valid_norm = pred_valid.norm(dim=0)
    target_valid_norm = target_valid.norm(dim=0)
    cosine = (pred_valid * target_valid).sum(dim=0) / (
        pred_valid_norm.clamp_min(1e-8) * target_valid_norm.clamp_min(1e-8)
    )
    sam = torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    sam = sam * (180.0 / math.pi)

    pred_spatial = pred_valid - pred_valid.mean()
    target_spatial = target_valid - target_valid.mean()
    spatial_cc = (pred_spatial * target_spatial).mean() / (
        pred_spatial.std(unbiased=False) * target_spatial.std(unbiased=False) + 1e-8
    )

    pred_spectral = pred_valid - pred_valid.mean(dim=0, keepdim=True)
    target_spectral = target_valid - target_valid.mean(dim=0, keepdim=True)
    spectral_cc = (pred_spectral * target_spectral).sum(dim=0) / (
        pred_spectral.square().sum(dim=0).sqrt().clamp_min(1e-8)
        * target_spectral.square().sum(dim=0).sqrt().clamp_min(1e-8)
    )

    result = {
        "hsi_psnr": psnr_from_mse(full_mse),
        "hsi_all_pixel_psnr": psnr_from_mse(full_mse),
        "hsi_foreground_psnr": psnr_from_mse(valid_mse),
        "hsi_rmse": float(full_mse.sqrt().item()),
        "hsi_foreground_rmse": float(valid_mse.sqrt().item()),
        "hsi_mae": float(full_error.abs().mean().item()),
        "hsi_foreground_mae": float(valid_error.abs().mean().item()),
        "hsi_sam_deg": float(sam.mean().item()),
        "hsi_sam_p95_deg": float(torch.quantile(sam, 0.95).item()),
        "hsi_cc": float(spatial_cc.item()),
        "hsi_spectral_cc": float(spectral_cc.mean().item()),
        "hsi_spectral_cc_p05": float(torch.quantile(spectral_cc, 0.05).item()),
        "valid_pixel_coverage": float(valid.float().mean().item()),
        "prediction_zero_rate_all": float(
            (pred_norm <= float(min_norm)).float().mean().item()
        ),
        "prediction_zero_rate_valid": float(
            (pred_valid_norm <= float(min_norm)).float().mean().item()
        ),
    }

    if bool(background.any()):
        background_error = flat_pred[:, background] - flat_target[:, background]
        result.update(
            {
                "hsi_background_rmse": float(background_error.square().mean().sqrt().item()),
                "hsi_background_mae": float(background_error.abs().mean().item()),
                "background_false_positive_spectral_energy": float(
                    flat_pred[:, background].norm(dim=0).mean().item()
                ),
            }
        )
    else:
        result.update(
            {
                "hsi_background_rmse": 0.0,
                "hsi_background_mae": 0.0,
                "background_false_positive_spectral_energy": 0.0,
            }
        )

    if pred.shape[0] > 1:
        derivative_error = (pred[1:] - pred[:-1]) - (target[1:] - target[:-1])
        result["hsi_derivative_rmse"] = float(derivative_error.square().mean().sqrt().item())
    else:
        result["hsi_derivative_rmse"] = 0.0
    return result
