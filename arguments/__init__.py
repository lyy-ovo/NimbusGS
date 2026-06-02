#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import os
import sys
import yaml

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            arg_name = key
            if arg_name.startswith("_"):
                shorthand = True
                arg_name = arg_name[1:]
            value_type = type(value)
            default_value = value if not fill_none else None
            if shorthand:
                if value_type == bool:
                    group.add_argument("--" + arg_name, "-" + arg_name[0:1], default=default_value, action="store_true")
                else:
                    group.add_argument("--" + arg_name, "-" + arg_name[0:1], default=default_value, type=value_type)
            else:
                if value_type == bool:
                    group.add_argument("--" + arg_name, default=default_value, action="store_true")
                else:
                    group.add_argument("--" + arg_name, default=default_value, type=value_type)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or "_" + key in vars(self):
                setattr(group, key, value)
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Dataset Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path) if group.source_path else ""
        return group

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.4
        self.densification_interval = 100
        self.opacity_reset_interval = 30000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.densify_max_clone_points = 60000
        self.densify_max_split_points = 60000
        self.densify_max_total_points = 1500000
        self.densify_split_factor = 2

        self.enable_fog = True
        self.fog_resolution = 128
        self.fog_lr = 0.005
        self.airlight_lr = 0.005
        self.lambda_tv = 1.0
        self.lambda_dcp = 1.0
        self.lambda_tau = 0.0
        self.density_scale_init = 4.0
        self.fog_auto_normalize = True
        self.use_suggested_density = True

        self.use_airlight_cnn = False
        self.use_airlight_cnn2 = False
        self.airlight_feature_dim = 64
        self.airlight_cnn_lr = 0.0001
        self.airlight_sh_degree = 0

        self.enable_rain = False
        self.derain_warmup_iters = 2000
        self.rain_diff_thr = 0.05
        self.rain_update_after_warmup = False
        self.rain_update_interval = 200
        self.densify_rain_mask_fill_ratio = 1.0
        self.densify_rain_mask_threshold = 0.5

        self.gt_path = ""
        self.best_metric = "PSNR"

        super().__init__(parser, "Optimization Parameters")

def _flatten_config(cfg):
    flat = {}
    for section in ("dataset", "pipeline", "optimization"):
        values = cfg.get(section, {})
        if values is not None:
            flat.update(values)
    return flat

def _load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def _build_parser():
    parser = ArgumentParser(description="NimbusGS training")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default="")
    return parser

def get_combined_args(parser):
    cmd_args = parser.parse_args(sys.argv[1:])
    cfg = _flatten_config(_load_yaml_config(getattr(cmd_args, "config", "")))

    for key, value in cfg.items():
        if hasattr(cmd_args, key):
            current = getattr(cmd_args, key)
            if current == parser.get_default(key):
                setattr(cmd_args, key, value)
        else:
            setattr(cmd_args, key, value)

    return cmd_args

def _ensure_config_argument(parser):
    for action in parser._actions:
        if "--config" in action.option_strings:
            return
    parser.add_argument("--config", type=str, default="")

def parse_configured_args(parser):
    _ensure_config_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    cfg = _flatten_config(_load_yaml_config(getattr(args, "config", "")))
    for key, value in cfg.items():
        if hasattr(args, key):
            default_value = parser.get_default(key)
            if getattr(args, key) == default_value:
                setattr(args, key, value)
        else:
            setattr(args, key, value)
    return args

def create_parser():
    parser = _build_parser()
    ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    return parser
