import json
import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, parse_configured_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from nimbusgs.metrics import evaluate_cameras

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

def main():
    parser = ArgumentParser(description="Evaluate NimbusGS renderings against a GT directory")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--render_image", choices=["render"], default="render")
    args = parse_configured_args(parser)

    dataset = model.extract(args)
    pipeline = pipe.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()

    def render_fn(cam):
        return render(cam, gaussians, pipeline, background, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]

    out_path = args.output_json or os.path.join(dataset.model_path, "metrics", f"{args.split}_metrics.json")
    result = evaluate_cameras(cameras, render_fn, args.gt_path, output_json=out_path)
    if result is None:
        raise RuntimeError(f"No matching GT images found in: {args.gt_path}")
    print(json.dumps(result["average"], indent=2))
    print(f"Saved metrics to: {out_path}")

if __name__ == "__main__":
    main()
