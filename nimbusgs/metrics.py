import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from utils.image_utils import psnr
from utils.loss_utils import ssim

try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")

def _find_gt(gt_dir: str, image_name: str) -> Optional[str]:
    root = Path(gt_dir)
    candidates = [root / image_name]
    stem = Path(image_name).stem
    candidates.extend(root / f"{stem}{ext}" for ext in _IMAGE_EXTS)
    for path in candidates:
        if path.exists():
            return str(path)
    return None

def _to_4d(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    return image

def load_gt_tensor(path: str, device: str = "cuda") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img).unsqueeze(0).to(device)

@torch.no_grad()
def compute_image_metrics(rendered: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    rendered = _to_4d(rendered).clamp(0, 1)
    gt = _to_4d(gt).clamp(0, 1)
    if gt.shape[-2:] != rendered.shape[-2:]:
        gt = torch.nn.functional.interpolate(gt, size=rendered.shape[-2:], mode="bilinear", align_corners=False)
    out = {
        "PSNR": float(psnr(rendered, gt).mean().item()),
        "SSIM": float(ssim(rendered.squeeze(0), gt.squeeze(0)).mean().item()),
    }
    if LPIPS_AVAILABLE:
        out["LPIPS"] = float(lpips(rendered, gt, net_type="vgg").mean().item())
    return out

@torch.no_grad()
def evaluate_cameras(cameras: Iterable, render_fn, gt_dir: str, output_json: Optional[str] = None) -> Optional[Dict]:
    per_image: Dict[str, Dict[str, float]] = {}
    for cam in cameras:
        gt_path = _find_gt(gt_dir, cam.image_name)
        if gt_path is None:
            continue
        rendered = render_fn(cam)
        gt = load_gt_tensor(gt_path, device=rendered.device).squeeze(0)
        per_image[cam.image_name] = compute_image_metrics(rendered, gt)

    if not per_image:
        return None

    keys = sorted(next(iter(per_image.values())).keys())
    average = {k: float(sum(v[k] for v in per_image.values()) / len(per_image)) for k in keys}
    result = {"average": average, "per_image": per_image, "num_images": len(per_image)}
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
