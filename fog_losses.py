import torch
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from dark_channel_prior import dark_channel_prior
import os
from PIL import Image
import torchvision.transforms as transforms

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

class FogLosses:

    def __init__(self, opt, source_path=None):
        self.opt = opt
        self.epsilon = 1e-2
        self.source_path = source_path
        self.mask_cache = {}

        self.lambda_tv = getattr(opt, 'lambda_tv', 0.05)
        self.derain_warmup_iters = getattr(opt, 'derain_warmup_iters', 2000)

    def _load_mask_image(self, viewpoint_cam):

        if self.source_path is None:
            return None

        cache_key = viewpoint_cam.image_name
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        mask_dir = os.path.join(self.source_path, "mask")
        if not os.path.exists(mask_dir):

            return None

        base_name = viewpoint_cam.image_name
        if '.' in base_name:
            base_name = os.path.splitext(base_name)[0]

        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                try:

                    mask_image = Image.open(mask_path).convert('L')
                    transform = transforms.Compose([
                        transforms.ToTensor()
                    ])
                    mask_tensor = transform(mask_image).squeeze(0)

                    mask_tensor = (mask_tensor > 0.5).float()

                    self.mask_cache[cache_key] = mask_tensor
                    return mask_tensor
                except Exception as e:
                    print(f"Failed to load mask image {mask_path}: {e}")
                    return None

        print(f"Mask image not found for {viewpoint_cam.image_name}")
        print(f"Looking for base name: {base_name}")
        print(f"Available files in {mask_dir}:")
        try:
            files = os.listdir(mask_dir)[:10]
            print(f"  {files}")
        except:
            print("  Could not list directory")

        return None

    def _dark_channel_prior_loss(self, clear_image, mask=None, patch_size=15, use_soft_min=True, temperature=0.1):

        clear_image_clamped = torch.clamp(clear_image, 0.0, 1.0)

        dark_channel = dark_channel_prior(clear_image_clamped, patch_size=patch_size,
                                         use_soft_min=use_soft_min, temperature=temperature)

        dark_channel = torch.clamp(dark_channel, min=0.0)

        if mask is not None:

            if mask.shape != dark_channel.shape:

                target_size = dark_channel.shape[-2:]
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                   size=target_size,
                                   mode='nearest').squeeze()

            masked_dark_channel = dark_channel * mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:

                dcp_mean = masked_dark_channel.sum() / valid_pixels
            else:
                dcp_mean = torch.tensor(0.0, device=clear_image.device)
        else:

            dcp_mean = torch.mean(dark_channel)

        dcp_loss = dcp_mean ** 2

        dcp_loss = dcp_loss + 1e-6

        return dcp_loss

    def compute_losses(self, clear_image, fogged_image, transmission, airlight, gt_fogged_image,
                      fog_field, airlight_field, iteration, depth_map=None, viewpoint_cam=None, scene=None):

        if iteration < 0:

            loss_rec1 = self._reconstruction_loss(clear_image, gt_fogged_image.to(clear_image.device))

            total_loss =  loss_rec1
            return {
                'total_loss': total_loss,

                'loss_rec1': loss_rec1
            }
        elif iteration < self.derain_warmup_iters:

            if depth_map is None or viewpoint_cam is None or not hasattr(airlight_field, 'eval_sh'):
                raise ValueError("Phase 2雾渲染需要 depth_map, viewpoint_cam, airlight_field.eval_sh")

            if depth_map.dim() == 3:
                if depth_map.shape[0] == 1:
                    depth_map = depth_map.squeeze(0)
                else:
                    depth_map = depth_map[0]
            elif depth_map.dim() != 2:
                raise ValueError(f"Unexpected depth_map shape: {depth_map.shape}")

            from fog_field import get_rays
            H, W = depth_map.shape
            ray_origins, ray_directions, dirs_camera_unit = get_rays(viewpoint_cam, H, W)
            z_depths = 1.0 / depth_map.clamp_min(1e-6)
            if dirs_camera_unit is not None:
                cos_theta = torch.abs(dirs_camera_unit[..., 2]).clamp_min(1e-3)
                surface_path_lengths = (z_depths / cos_theta).clamp_min(1e-6)
            else:
                surface_path_lengths = z_depths.clamp_min(1e-6)
            world_points = ray_origins + ray_directions * surface_path_lengths.unsqueeze(-1)
            airlight_SH = airlight_field.eval_sh
            C_haze, transmission = fog_field.render_haze_color(
                ray_origins, ray_directions, depth_map, dirs_camera_unit, airlight_SH, num_samples=32)
            if transmission.dim() == 2:
                transmission = transmission.unsqueeze(0)
            t_expanded = transmission.expand_as(clear_image)
            fogged_synthetic_raw = clear_image * t_expanded + C_haze.permute(2,0,1)

            fogged_synthetic = torch.clamp(fogged_synthetic_raw, 0.0, 1.0)

            mask = self._load_mask_image(viewpoint_cam)
            if mask is not None:

                if mask.shape != clear_image.shape[-2:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                       size=clear_image.shape[-2:],
                                       mode='nearest').squeeze()
                mask = mask.to(clear_image.device)

            loss_rec = self._reconstruction_loss(fogged_synthetic, gt_fogged_image)

            clear_image_normalized = (clear_image.clamp(-1, 1) + 1) / 2

            loss_dcp = self._dark_channel_prior_loss(clear_image_normalized, mask=mask)

            loss_tau_prior = self._tau_prior(transmission.squeeze(0) if transmission.dim() == 3 else transmission, iteration)

            loss_tv = fog_field.voxel_grid.compute_tv_loss()
            lambda_dc = self.opt.lambda_dcp if hasattr(self.opt, 'lambda_dcp') else 1.0
            lambda_tau = self.opt.lambda_tau if hasattr(self.opt, 'lambda_tau') else 0.1

            total_loss = (loss_rec +
                         lambda_dc * loss_dcp +
                         lambda_tau * loss_tau_prior +
                         self.lambda_tv * loss_tv
                         )
            airlight_raw = airlight_SH(dirs_camera_unit)
            airlight_colors = (torch.tanh(airlight_raw) + 1.0) * 0.5

            return {
                'total_loss': total_loss,
                'loss_rec': loss_rec,
                'loss_dcp': loss_dcp,
                'loss_tau_prior': loss_tau_prior,
                'loss_tv': loss_tv,
                'C_haze': C_haze,
                'transmission': transmission,
                'fogged_synthetic': fogged_synthetic,
                'clear_image': clear_image,
                'airlight_colors': airlight_colors,
                'airlight_colors_raw': airlight_raw,
                'mask': mask
            }

        else:

            if depth_map is None or viewpoint_cam is None or not hasattr(airlight_field, 'eval_sh'):
                raise ValueError("Phase 2雾渲染需要 depth_map, viewpoint_cam, airlight_field.eval_sh")

            if depth_map.dim() == 3:
                if depth_map.shape[0] == 1:
                    depth_map = depth_map.squeeze(0)
                else:
                    depth_map = depth_map[0]
            elif depth_map.dim() != 2:
                raise ValueError(f"Unexpected depth_map shape: {depth_map.shape}")

            from fog_field import get_rays
            H, W = depth_map.shape
            ray_origins, ray_directions, dirs_camera_unit = get_rays(viewpoint_cam, H, W)
            z_depths = 1.0 / depth_map.clamp_min(1e-6)
            if dirs_camera_unit is not None:
                cos_theta = torch.abs(dirs_camera_unit[..., 2]).clamp_min(1e-3)
                surface_path_lengths = (z_depths / cos_theta).clamp_min(1e-6)
            else:
                surface_path_lengths = z_depths.clamp_min(1e-6)
            world_points = ray_origins + ray_directions * surface_path_lengths.unsqueeze(-1)
            airlight_SH = airlight_field.eval_sh
            C_haze, transmission = fog_field.render_haze_color(
                ray_origins, ray_directions, depth_map, dirs_camera_unit, airlight_SH, num_samples=32)
            if transmission.dim() == 2:
                transmission = transmission.unsqueeze(0)
            t_expanded = transmission.expand_as(clear_image)
            fogged_synthetic_raw = clear_image * t_expanded + C_haze.permute(2,0,1)

            fogged_synthetic = torch.clamp(fogged_synthetic_raw, 0.0, 1.0)

            mask = self._load_mask_image(viewpoint_cam)
            if mask is not None:

                if mask.shape != clear_image.shape[-2:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                       size=clear_image.shape[-2:],
                                       mode='nearest').squeeze()
                mask = mask.to(clear_image.device)

            loss_rec = self._reconstruction_loss(fogged_synthetic, gt_fogged_image)

            clear_image_normalized = (clear_image.clamp(-1, 1) + 1) / 2

            loss_dcp = self._dark_channel_prior_loss(clear_image_normalized, mask=mask)

            lambda_dc = self.opt.lambda_dcp if hasattr(self.opt, 'lambda_dcp') else 1.0
            lambda_tau = self.opt.lambda_tau if hasattr(self.opt, 'lambda_tau') else 0.0
            loss_tau_prior = self._tau_prior(transmission.squeeze(0) if transmission.dim() == 3 else transmission, iteration)

            loss_tv = fog_field.voxel_grid.compute_tv_loss()

            total_loss = (loss_rec +
                         lambda_dc * loss_dcp +
                         lambda_tau * loss_tau_prior +
                         self.lambda_tv * loss_tv)

            airlight_raw = airlight_SH(dirs_camera_unit)
            airlight_colors = (torch.tanh(airlight_raw) + 1.0) * 0.5

            return {
                'total_loss': total_loss,
                'loss_rec': loss_rec,
                'loss_dcp': loss_dcp,
                'loss_tau_prior': loss_tau_prior,
                'loss_tau': loss_tau_prior,
                'loss_tv': loss_tv,

                'C_haze': C_haze,
                'transmission': transmission,
                'fogged_synthetic': fogged_synthetic,
                'clear_image': clear_image,
                'airlight_colors': airlight_colors,
                'airlight_colors_raw': airlight_raw,
                'mask': mask
            }

    def _tau_prior(self, transmission, iteration):

        iters = self.opt.iterations
        progress = min(iteration / (0.5 * iters), 1.0)
        tau_target = 0.05 + 0.20 * progress

        tau = 1.0 - transmission
        if tau.ndim == 3:
            tau = tau.mean(dim=0)

        tau_mean = tau.mean()

        high_trans_penalty = torch.mean(torch.relu(transmission - 0.95) ** 2)

        return (tau_mean - tau_target) ** 2 + 0.1 * high_trans_penalty

    def _reconstruction_loss(self, pred_fogged, gt_fogged):

        l1 = l1_loss(pred_fogged, gt_fogged)
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(pred_fogged.unsqueeze(0), gt_fogged.unsqueeze(0))
        else:
            ssim_val = ssim(pred_fogged, gt_fogged)
        return (1.0 - self.opt.lambda_dssim)* l1 + self.opt.lambda_dssim * (1.0 - ssim_val)
