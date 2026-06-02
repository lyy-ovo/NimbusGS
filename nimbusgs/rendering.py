import torch

from fog_field import get_rays

@torch.no_grad()
def synthesize_fogged_image(camera, clear_image, depth, fog_field, airlight_field, num_samples=32):
    if depth.dim() == 3:
        depth = depth.squeeze(0) if depth.shape[0] == 1 else depth[0]
    h, w = depth.shape
    ray_o, ray_d, dirs_cam = get_rays(camera, h, w)

    if getattr(airlight_field, "use_cnn", False):
        transmission_hw = fog_field.compute_transmission(ray_o, ray_d, depth, dirs_cam, num_samples=num_samples)
        airlight_raw = airlight_field(positions=dirs_cam, view_directions=dirs_cam,
                                      clear_image=clear_image, transmission=transmission_hw)
        airlight = (torch.tanh(airlight_raw) + 1.0) * 0.5
        c_haze = airlight * (1.0 - transmission_hw.unsqueeze(-1))
        transmission = transmission_hw.unsqueeze(0)
    else:
        c_haze, transmission = fog_field.render_haze_color(ray_o, ray_d, depth, dirs_cam,
                                                           airlight_field.eval_sh, num_samples=num_samples)
        if transmission.dim() == 2:
            transmission = transmission.unsqueeze(0)
        airlight = (torch.tanh(airlight_field.eval_sh(dirs_cam)) + 1.0) * 0.5

    fogged = (clear_image * transmission.expand_as(clear_image) + c_haze.permute(2, 0, 1)).clamp(0, 1)
    return {
        "fogged": fogged,
        "clear": clear_image,
        "transmission": transmission,
        "C_haze": c_haze,
        "airlight": airlight,
    }
