import torch
import torch.nn.functional as F

def dark_channel_prior(image, patch_size=15, use_soft_min=False, temperature=0.1):

    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, C, H, W = image.shape

    if use_soft_min:

        min_rgb = -temperature * torch.logsumexp(-image / temperature, dim=1, keepdim=True)
    else:
        min_rgb = torch.min(image, dim=1, keepdim=True)[0]

    padding = patch_size // 2
    min_rgb_padded = F.pad(min_rgb, (padding, padding, padding, padding), mode='reflect')

    patches = F.unfold(min_rgb_padded, kernel_size=patch_size, stride=1)
    patches = patches.view(B, patch_size * patch_size, H, W)

    if use_soft_min:

        dark_channel = -temperature * torch.logsumexp(-patches / temperature, dim=1, keepdim=True)
    else:
        dark_channel = torch.min(patches, dim=1, keepdim=True)[0]

    if squeeze_output:
        dark_channel = dark_channel.squeeze(0)

    return dark_channel

def estimate_atmospheric_light_dcp(image, dark_channel=None, top_percent=0.1, use_soft_min=False, temperature=0.1):

    if image.dim() == 3:
        image = image.unsqueeze(0)
        if dark_channel is not None:
            dark_channel = dark_channel.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if dark_channel is None:
        dark_channel = dark_channel_prior(image, use_soft_min=use_soft_min, temperature=temperature)

    B, C, H, W = image.shape

    atmospheric_lights = []

    for b in range(B):
        img_b = image[b]
        dark_b = dark_channel[b, 0]

        flat_dark = dark_b.flatten()
        num_pixels = flat_dark.numel()
        num_top = max(1, int(num_pixels * top_percent))

        _, top_indices = torch.topk(flat_dark, num_top)

        top_intensities = []
        for idx in top_indices:
            y, x = torch.div(idx, W, rounding_mode='floor'), idx % W
            pixel_intensity = img_b[:, y, x].sum()
            top_intensities.append((pixel_intensity, img_b[:, y, x]))

        _, atmospheric_light = max(top_intensities, key=lambda x: x[0])
        atmospheric_lights.append(atmospheric_light)

    atmospheric_lights = torch.stack(atmospheric_lights)

    if squeeze_output:
        atmospheric_lights = atmospheric_lights.squeeze(0)

    return atmospheric_lights

def transmission_from_dcp(image, atmospheric_light, omega=0.95, patch_size=15, use_soft_min=False, temperature=0.1):

    if image.dim() == 3:
        image = image.unsqueeze(0)
        atmospheric_light = atmospheric_light.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, C, H, W = image.shape

    atmospheric_light = atmospheric_light.view(B, C, 1, 1)
    normalized_image = image / (atmospheric_light + 1e-6)

    dark_channel_norm = dark_channel_prior(normalized_image, patch_size, use_soft_min=use_soft_min, temperature=temperature)

    transmission = 1 - omega * dark_channel_norm

    transmission = torch.clamp(transmission, min=0.1, max=1.0)

    if squeeze_output:
        transmission = transmission.squeeze(0)

    return transmission

def depth_dependent_transmission(depth_map, scattering_coeff=0.1, max_depth=50.0):

    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(0)

    depth_normalized = torch.clamp(depth_map / max_depth, 0, 1)

    transmission = torch.exp(-scattering_coeff * depth_normalized * max_depth)

    return transmission

def combined_transmission_prior(fogged_image, depth_map, prev_transmission=None,
                               alpha_dcp=0.3, alpha_depth=0.4, alpha_prev=0.3,
                               use_soft_min=False, temperature=0.1):

    device = fogged_image.device

    atmospheric_light = estimate_atmospheric_light_dcp(fogged_image,
                                                      None, use_soft_min=use_soft_min, temperature=temperature)
    transmission_dcp = transmission_from_dcp(fogged_image, atmospheric_light,
                                            use_soft_min=use_soft_min, temperature=temperature)

    transmission_depth = depth_dependent_transmission(depth_map)

    combined = alpha_dcp * transmission_dcp + alpha_depth * transmission_depth

    if prev_transmission is not None:
        combined = alpha_dcp * transmission_dcp + alpha_depth * transmission_depth + alpha_prev * prev_transmission
        combined = combined / (alpha_dcp + alpha_depth + alpha_prev)
    else:
        combined = combined / (alpha_dcp + alpha_depth)

    combined = torch.clamp(combined, min=0.1, max=0.95)

    return combined, transmission_dcp, transmission_depth

if __name__ == "__main__":

    print("测试暗通道先验实现...")

    test_image = torch.rand(3, 256, 256) * 0.8 + 0.1
    test_depth = torch.rand(256, 256) * 20 + 5

    dark_ch_hard = dark_channel_prior(test_image, use_soft_min=False)
    print(f"硬暗通道形状: {dark_ch_hard.shape}")
    print(f"硬暗通道范围: [{dark_ch_hard.min():.4f}, {dark_ch_hard.max():.4f}]")

    dark_ch_soft = dark_channel_prior(test_image, use_soft_min=True, temperature=0.1)
    print(f"软暗通道形状: {dark_ch_soft.shape}")
    print(f"软暗通道范围: [{dark_ch_soft.min():.4f}, {dark_ch_soft.max():.4f}]")

    atm_light = estimate_atmospheric_light_dcp(test_image, dark_ch_soft, use_soft_min=True)
    print(f"大气光: {atm_light}")

    trans_dcp = transmission_from_dcp(test_image, atm_light, use_soft_min=True)
    print(f"DCP透射率范围: [{trans_dcp.min():.4f}, {trans_dcp.max():.4f}]")

    trans_depth = depth_dependent_transmission(test_depth)
    print(f"深度透射率范围: [{trans_depth.min():.4f}, {trans_depth.max():.4f}]")

    combined_trans, _, _ = combined_transmission_prior(test_image, test_depth, use_soft_min=True)
    print(f"组合透射率范围: [{combined_trans.min():.4f}, {combined_trans.max():.4f}]")

    print("✅ DCP先验测试完成！")
    print("   - 软min pooling已启用，平滑且可导")
    print("   - 硬min和软min对比完成")
