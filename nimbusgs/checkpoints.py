import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

@dataclass
class BestCheckpointState:
    metric_name: str = "PSNR"
    higher_is_better: bool = True
    best_value: Optional[float] = None
    best_iteration: Optional[int] = None

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.higher_is_better else value < self.best_value

def checkpoint_payload(iteration: int, gaussians, fog_field=None, airlight_field=None, extra: Optional[Dict[str, Any]] = None):
    payload = {
        "iteration": int(iteration),
        "gaussians": gaussians.capture(),
        "extra": extra or {},
    }
    if fog_field is not None:
        payload["fog_field"] = {k: v.detach().cpu() for k, v in fog_field.state_dict().items()}
    if airlight_field is not None:
        payload["airlight_field"] = {k: v.detach().cpu() for k, v in airlight_field.state_dict().items()}
    return payload

def save_checkpoint(path: str, iteration: int, gaussians, fog_field=None, airlight_field=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint_payload(iteration, gaussians, fog_field, airlight_field, extra), path)

def save_best_checkpoint(output_dir: str, iteration: int, metric_value: float, state: BestCheckpointState,
                         gaussians, fog_field=None, airlight_field=None, metrics: Optional[Dict[str, Any]] = None):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if not state.is_better(float(metric_value)):
        return False

    state.best_value = float(metric_value)
    state.best_iteration = int(iteration)
    best_path = os.path.join(ckpt_dir, "best.pth")
    save_checkpoint(best_path, iteration, gaussians, fog_field, airlight_field,
                    extra={"best_metric": state.metric_name, "best_value": state.best_value, "metrics": metrics or {}})

    summary = {
        "best_iteration": state.best_iteration,
        "best_metric": state.metric_name,
        "best_value": state.best_value,
    }
    with open(os.path.join(ckpt_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return True

def load_best_checkpoint(output_dir: str, map_location="cuda"):
    best_path = os.path.join(output_dir, "checkpoints", "best.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(best_path)
    return torch.load(best_path, map_location=map_location)
