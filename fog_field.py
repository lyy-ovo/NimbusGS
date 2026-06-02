import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def tv_loss_3d(voxel_grid):

    dx = voxel_grid[:, :, 1:, :, :] - voxel_grid[:, :, :-1, :, :]
    dy = voxel_grid[:, :, :, 1:, :] - voxel_grid[:, :, :, :-1, :]
    dz = voxel_grid[:, :, :, :, 1:] - voxel_grid[:, :, :, :, :-1]

    tv_loss = dx.abs().mean() + dy.abs().mean() + dz.abs().mean()

    return tv_loss

def ray_aabb_intersection(ray_origins, ray_directions, aabb_min, aabb_max):

    device = ray_origins.device
    H, W = ray_origins.shape[:2]

    aabb_min = torch.tensor(aabb_min, device=device, dtype=torch.float32)
    aabb_max = torch.tensor(aabb_max, device=device, dtype=torch.float32)

    ray_directions = ray_directions.clone()
    eps = 1e-8
    ray_directions = torch.where(torch.abs(ray_directions) < eps,
                                torch.sign(ray_directions) * eps,
                                ray_directions)

    inv_dir = 1.0 / ray_directions

    t1 = (aabb_min[None, None, :] - ray_origins) * inv_dir
    t2 = (aabb_max[None, None, :] - ray_origins) * inv_dir

    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    t_entry = torch.max(t_min, dim=2)[0]

    t_exit = torch.min(t_max, dim=2)[0]

    valid_mask = (t_entry <= t_exit) & (t_exit > 0)

    t_entry = torch.where(valid_mask, t_entry, torch.tensor(-1.0, device=device))
    t_exit = torch.where(valid_mask, t_exit, torch.tensor(-1.0, device=device))

    t_entry = torch.clamp(t_entry, min=0.0)

    return t_entry, t_exit, valid_mask

class VoxelGrid(nn.Module):

    def __init__(self, bbox_min, bbox_max, resolution=128, density_scale_init=0.3, world_to_grid_transform=None):
        super().__init__()

        self.register_buffer('bbox_min', torch.tensor(bbox_min, dtype=torch.float32))
        self.register_buffer('bbox_max', torch.tensor(bbox_max, dtype=torch.float32))
        self.resolution = resolution

        if world_to_grid_transform is not None:
            self.register_buffer('world_center', torch.tensor(world_to_grid_transform['center'], dtype=torch.float32))
            self.register_buffer('world_scale', torch.tensor(world_to_grid_transform['scale'], dtype=torch.float32))
            self.use_world_transform = True
        else:
            self.use_world_transform = False

        self.voxel_features = nn.Parameter(
            torch.randn(1, 1, resolution, resolution, resolution) * 0.01
        )

        self.density_scale_log = nn.Parameter(
            torch.log(torch.tensor(density_scale_init, dtype=torch.float32))
        )

    def forward(self, xyz):

        if self.use_world_transform:

            normalized_xyz = (xyz - self.world_center) / self.world_scale
        else:

            normalized_xyz = 2.0 * (xyz - self.bbox_min) / (
                self.bbox_max - self.bbox_min
            ) - 1.0

        in_bounds = torch.all((normalized_xyz >= -1.0) & (normalized_xyz <= 1.0), dim=-1)

        if self.use_world_transform:
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0

        in_bounds = torch.all((normalized_xyz >= -1.0) & (normalized_xyz <= 1.0), dim=-1)

        sample_points = normalized_xyz.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        features = F.grid_sample(
            self.voxel_features,
            sample_points,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        density_scale = torch.exp(self.density_scale_log)
        beta = F.softplus(features.squeeze()) * density_scale

        if beta.dim() == 0:
            beta = beta.unsqueeze(0)

        beta = beta * in_bounds.float()

        return beta

    def get_density_scale(self):

        return torch.exp(self.density_scale_log).item()

    def set_density_scale(self, scale):

        with torch.no_grad():
            self.density_scale_log.copy_(torch.log(torch.tensor(scale, dtype=torch.float32)))

    def compute_tv_loss(self):

        beta_grid = torch.nn.functional.softplus(self.voxel_features) * torch.exp(self.density_scale_log)
        return tv_loss_3d(beta_grid)

class FogTransmissionField(nn.Module):
    def render_haze_color(self, ray_origins, ray_directions, depths, dirs_camera_unit, airlight_SH, num_samples=64):

        if depths.dim() == 3:
            if depths.shape[0] == 1:
                depths = depths.squeeze(0)
            else:
                depths = depths[0]
        H, W = depths.shape
        device = depths.device

        z_depths = 1.0 / depths.clamp_min(1e-6)

        if dirs_camera_unit is not None:
            cos_theta = torch.abs(dirs_camera_unit[..., 2]).clamp_min(1e-3)
            surface_path_lengths = z_depths / cos_theta
        else:
            surface_path_lengths = z_depths
        surface_path_lengths = surface_path_lengths.clamp_min(1e-6)

        aabb_min = self.bbox_min
        aabb_max = self.bbox_max
        t_entry, t_exit, valid_mask = ray_aabb_intersection(ray_origins, ray_directions, aabb_min, aabb_max)

        s_min = t_entry.clamp_min(0.0)
        s_max = torch.minimum(t_exit, surface_path_lengths)

        valid_sampling = valid_mask & (s_min < s_max) & (s_max > 0)

        C_haze = torch.zeros((H, W, 3), device=device)
        transmission = torch.ones((H, W), device=device)

        if not bool(valid_sampling.any()):
            return C_haze, transmission

        valid_indices = torch.where(valid_sampling)
        num_valid = valid_indices[0].numel()
        if num_valid == 0:
            return C_haze, transmission

        valid_ray_origins = ray_origins[valid_indices]
        valid_ray_directions = ray_directions[valid_indices]
        valid_s_min = s_min[valid_indices]
        valid_s_max = s_max[valid_indices]

        sample_positions = self._get_sample_positions(num_samples, device)
        s_vals = valid_s_min.unsqueeze(-1) + (valid_s_max - valid_s_min).unsqueeze(-1) * sample_positions.unsqueeze(0)

        sample_points = valid_ray_origins.unsqueeze(1) + valid_ray_directions.unsqueeze(1) * s_vals.unsqueeze(-1)
        beta_values = self.voxel_grid(sample_points.reshape(-1, 3)).view(num_valid, num_samples)

        airlight_raw = airlight_SH(valid_ray_directions)
        airlight = (torch.tanh(airlight_raw) + 1.0) * 0.5
        airlight_samples = airlight.unsqueeze(1).expand(-1, num_samples, -1)

        segment_lengths = ((valid_s_max - valid_s_min) / num_samples).unsqueeze(-1)
        optical_depths = beta_values * segment_lengths
        cumulative_optical_depth = torch.cumsum(optical_depths, dim=1)
        exclusive_cumulative = torch.cat([
            torch.zeros((num_valid, 1), device=device),
            cumulative_optical_depth[:, :-1]
        ], dim=1)
        T_samples = torch.exp(-exclusive_cumulative)

        segment_contribution = -torch.expm1(-optical_depths)
        haze_integrand = airlight_samples * T_samples.unsqueeze(-1) * segment_contribution.unsqueeze(-1)
        valid_C_haze = haze_integrand.sum(dim=1)

        valid_transmission = torch.exp(-cumulative_optical_depth[:, -1])

        C_haze[valid_indices] = valid_C_haze
        transmission[valid_indices] = valid_transmission

        return C_haze, transmission

    def __init__(self, scene_bbox=None, gaussians=None, resolution=128, density_scale_init=0.3, auto_normalize=True):
        super().__init__()

        self.original_bbox_min = None
        self.original_bbox_max = None
        world_to_grid_transform = None

        if scene_bbox is None and gaussians is not None:
            bbox_min, bbox_max, world_to_grid_transform = self._compute_scene_bbox(gaussians, auto_normalize)
        elif scene_bbox is not None:
            bbox_min, bbox_max = scene_bbox
        else:

            bbox_min, bbox_max = [-1, -1, -1], [1, 1, 1]

        self.voxel_grid = VoxelGrid(bbox_min, bbox_max, resolution, density_scale_init, world_to_grid_transform)

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.auto_normalize = auto_normalize
        self._sample_cache = {}

    def _compute_scene_bbox(self, gaussians, auto_normalize):

        xyz_points = gaussians.get_xyz.detach().cpu().numpy()
        bbox_min = xyz_points.min(axis=0)
        bbox_max = xyz_points.max(axis=0)

        scene_size = bbox_max - bbox_min
        expansion_factor = 2.0
        expanded_size = scene_size * expansion_factor

        scene_center = (bbox_min + bbox_max) / 2
        bbox_min_expanded = scene_center - expanded_size / 2
        bbox_max_expanded = scene_center + expanded_size / 2

        self.original_bbox_min = bbox_min
        self.original_bbox_max = bbox_max

        world_to_grid_transform = None

        return bbox_min_expanded, bbox_max_expanded, world_to_grid_transform

    def compute_suggested_density_scale(self, avg_ray_length, target_transmission=0.5):

        target_optical_depth = -np.log(target_transmission)
        suggested_scale = target_optical_depth / avg_ray_length

        print(f"Suggested density scale: {suggested_scale:.6f}")
        print(f"(Target transmission {target_transmission} at ray length {avg_ray_length:.4f})")

        return suggested_scale

    def _get_sample_positions(self, num_samples, device):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        key = (device.type, device.index if device.index is not None else -1, num_samples)
        cached = self._sample_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.linspace(0.0, 1.0, num_samples, device=device)
            self._sample_cache[key] = cached
        return cached

    def _compute_segment_transmission(self, ray_origins, ray_directions, max_distances, num_samples=64):
        device = ray_origins.device
        num_rays = ray_origins.shape[0]

        transmissions = torch.ones(num_rays, device=device)
        if num_rays == 0:
            return transmissions, torch.zeros(num_rays, dtype=torch.bool, device=device)

        origins_hw = ray_origins.view(num_rays, 1, 3)
        directions_hw = ray_directions.view(num_rays, 1, 3)
        t_entry, t_exit, valid_mask = ray_aabb_intersection(origins_hw, directions_hw, self.bbox_min, self.bbox_max)
        t_entry = t_entry.view(num_rays)
        t_exit = t_exit.view(num_rays)
        valid_mask = valid_mask.view(num_rays)

        s_min = t_entry.clamp_min(0.0)
        s_max = torch.minimum(t_exit, max_distances)
        valid_sampling = valid_mask & (s_max > s_min) & (s_max > 0)

        if not bool(valid_sampling.any()):
            return transmissions, valid_sampling

        idx = torch.where(valid_sampling)[0]
        origins_valid = ray_origins[idx]
        directions_valid = ray_directions[idx]
        s_min_valid = s_min[idx]
        s_max_valid = s_max[idx]

        sample_positions = self._get_sample_positions(num_samples, device)
        s_vals = s_min_valid.unsqueeze(-1) + (s_max_valid - s_min_valid).unsqueeze(-1) * sample_positions.unsqueeze(0)
        sample_points = origins_valid.unsqueeze(1) + directions_valid.unsqueeze(1) * s_vals.unsqueeze(-1)

        beta_values = self.voxel_grid(sample_points.reshape(-1, 3)).view(-1, num_samples)
        segment_lengths = ((s_max_valid - s_min_valid) / num_samples).unsqueeze(-1)
        optical_depths = beta_values * segment_lengths
        cumulative_optical_depth = torch.cumsum(optical_depths, dim=1)
        transmissions_valid = torch.exp(-cumulative_optical_depth[:, -1]).clamp(0.0, 1.0)

        transmissions[idx] = transmissions_valid
        return transmissions, valid_sampling

    def compute_transmission_to_points(self, ray_origins, target_points, num_samples=64):

        device = ray_origins.device
        directions = target_points - ray_origins
        distances = torch.norm(directions, dim=-1)
        valid_distance = distances > 1e-6

        transmissions = torch.ones(ray_origins.shape[0], device=device)
        valid_mask = torch.zeros_like(valid_distance)

        if bool(valid_distance.any()):
            normalized_dirs = directions[valid_distance] / distances[valid_distance].unsqueeze(-1)
            subset_transmissions, subset_valid = self._compute_segment_transmission(
                ray_origins[valid_distance], normalized_dirs, distances[valid_distance], num_samples=num_samples
            )
            transmissions[valid_distance] = subset_transmissions
            valid_mask[valid_distance] = subset_valid

        return transmissions, valid_mask

    def compute_local_extinction_to_points(self, ray_origins, target_points, delta_s, num_samples=64, min_trans=1e-6):

        device = ray_origins.device
        directions = target_points - ray_origins
        s1 = torch.norm(directions, dim=-1)
        valid_distance = s1 > 1e-6

        dir_unit = torch.zeros_like(directions)
        if bool(valid_distance.any()):
            dir_unit[valid_distance] = directions[valid_distance] / s1[valid_distance].unsqueeze(-1)

        if not torch.is_tensor(delta_s):
            delta_s = torch.tensor(delta_s, device=device, dtype=s1.dtype)
        if delta_s.dim() == 0:
            delta_s = delta_s.expand_as(s1)
        delta_s = torch.clamp(delta_s, min=1e-6)
        s0 = torch.clamp(s1 - delta_s, min=1e-6)

        T1 = torch.ones_like(s1)
        T0 = torch.ones_like(s1)
        valid1 = torch.zeros_like(valid_distance)
        valid0 = torch.zeros_like(valid_distance)

        if bool(valid_distance.any()):
            T1_subset, valid1_subset = self._compute_segment_transmission(
                ray_origins[valid_distance], dir_unit[valid_distance], s1[valid_distance], num_samples=num_samples
            )
            T0_subset, valid0_subset = self._compute_segment_transmission(
                ray_origins[valid_distance], dir_unit[valid_distance], s0[valid_distance], num_samples=num_samples
            )
            T1[valid_distance] = T1_subset
            T0[valid_distance] = T0_subset
            valid1[valid_distance] = valid1_subset
            valid0[valid_distance] = valid0_subset

        eps = 1e-6
        T1_clamped = torch.clamp(T1, min=min_trans, max=1.0 - 1e-8)
        T0_clamped = torch.clamp(T0, min=min_trans, max=1.0 - 1e-8)
        sigma_loc = -(torch.log(T1_clamped) - torch.log(T0_clamped)) / torch.clamp(delta_s, min=eps)

        valid_mask = valid_distance & valid1 & valid0 & (s1 > s0)

        sigma_loc = torch.where(valid_mask, sigma_loc, torch.zeros_like(sigma_loc))
        return sigma_loc, valid_mask

    def get_density_scale(self):

        return self.voxel_grid.get_density_scale()

    def set_density_scale(self, scale):

        self.voxel_grid.set_density_scale(scale)

    def compute_transmission(self, ray_origins, ray_directions, depths, dirs_camera_unit=None, num_samples=64):

        if depths.dim() == 3:
            if depths.shape[0] == 1:
                depths = depths.squeeze(0)
            else:
                depths = depths[0]

        H, W = depths.shape
        device = depths.device

        z_depths = 1.0 / depths.clamp_min(1e-6)

        if hasattr(self, 'original_bbox_min') and self.original_bbox_min is not None:
            scene_diagonal = torch.norm(torch.tensor(self.original_bbox_max) - torch.tensor(self.original_bbox_min))
            max_ray_length = scene_diagonal.item() * 2.0

            z_depths = torch.clamp(z_depths, max=max_ray_length)

        if dirs_camera_unit is not None:

            cos_theta = torch.abs(dirs_camera_unit[..., 2])
            cos_theta = torch.clamp(cos_theta, min=1e-3)

            path_lengths = z_depths / cos_theta

        else:

            path_lengths = z_depths

        path_lengths = torch.clamp(path_lengths, min=1e-6)

        s_vals = torch.linspace(0, 1, num_samples, device=device)
        s_vals = s_vals[None, None, :].expand(H, W, num_samples)

        path_lengths_expanded = path_lengths.unsqueeze(-1)
        s_vals = s_vals * path_lengths_expanded

        ray_origins = ray_origins.unsqueeze(-2)
        ray_directions = ray_directions.unsqueeze(-2)
        s_vals_expanded = s_vals.unsqueeze(-1)

        sample_points = ray_origins + ray_directions * s_vals_expanded

        sample_points_flat = sample_points.reshape(-1, 3)

        if hasattr(self, 'original_bbox_min') and self.original_bbox_min is not None:
            bbox_min = torch.tensor(self.original_bbox_min, device=sample_points_flat.device)
            bbox_max = torch.tensor(self.original_bbox_max, device=sample_points_flat.device)

            in_bbox = torch.all(sample_points_flat >= bbox_min, dim=1) & torch.all(sample_points_flat <= bbox_max, dim=1)
            in_bbox_ratio = in_bbox.float().mean().item()

        beta_values = self.voxel_grid(sample_points_flat)
        beta_values = beta_values.reshape(H, W, num_samples)

        delta_s = path_lengths / (num_samples - 1)
        delta_s = delta_s.unsqueeze(-1).expand(-1, -1, num_samples - 1)

        beta_avg = 0.5 * (beta_values[:, :, 1:] + beta_values[:, :, :-1])

        optical_depth = torch.sum(beta_avg * delta_s, dim=-1)

        transmission = torch.exp(-optical_depth)

        return transmission

    def get_beta_values(self, xyz):

        return self.voxel_grid(xyz)

def get_rays(viewpoint_camera, H, W):

    device = viewpoint_camera.world_view_transform.device

    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device),
        torch.linspace(0, H-1, H, device=device),
        indexing='ij'
    )
    i = i.t()
    j = j.t()

    x = (i - W * 0.5) / (W * 0.5)
    y = (j - H * 0.5) / (H * 0.5)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    dirs_camera = torch.stack([
        x * tanfovx,
        -y * tanfovy,
        -torch.ones_like(x)
    ], dim=-1)

    dirs_camera_unit = dirs_camera / torch.norm(dirs_camera, dim=-1, keepdim=True)

    camera_to_world = torch.inverse(viewpoint_camera.world_view_transform).T
    rotation_matrix = camera_to_world[:3, :3]

    ray_directions = torch.sum(dirs_camera[..., None, :] * rotation_matrix, dim=-1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    ray_origins = viewpoint_camera.camera_center.expand(H, W, 3)

    return ray_origins, ray_directions, dirs_camera_unit
