import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LightweightCNN(nn.Module):

    def __init__(self, feature_dim=64, output_channels=3):
        super().__init__()

        in_channels = feature_dim + 1
        hidden_dim = 64

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(hidden_dim//2, hidden_dim//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim//4, output_channels, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim//2)
        self.bn4 = nn.BatchNorm2d(hidden_dim//4)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, degraded_features, transmission):

        H_feat, W_feat = degraded_features.shape[2], degraded_features.shape[3]
        transmission_down = F.interpolate(transmission, size=(H_feat, W_feat), mode='bilinear', align_corners=True)

        x = torch.cat([degraded_features, transmission_down], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)

        airlight = self.upsample(x)

        return airlight

class DegradationFeatureExtractor(nn.Module):

    def __init__(self, in_channels=3, feature_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(feature_dim)

    def forward(self, image):

        x = F.relu(self.bn1(self.conv1(image)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        return x

class SphericalHarmonics:

    @staticmethod
    def eval_sh(deg, sh, dirs):

        assert deg <= 2, "Only support SH degree <= 2"

        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        sh_basis = []

        sh_basis.append(0.5 * math.sqrt(1.0 / math.pi) * torch.ones_like(x))

        if deg >= 1:

            sh_basis.append(-0.5 * math.sqrt(3.0 / math.pi) * y)
            sh_basis.append(0.5 * math.sqrt(3.0 / math.pi) * z)
            sh_basis.append(-0.5 * math.sqrt(3.0 / math.pi) * x)

        if deg >= 2:

            sh_basis.append(0.5 * math.sqrt(15.0 / math.pi) * x * y)
            sh_basis.append(-0.5 * math.sqrt(15.0 / math.pi) * y * z)
            sh_basis.append(0.25 * math.sqrt(5.0 / math.pi) * (3 * z**2 - 1))
            sh_basis.append(-0.5 * math.sqrt(15.0 / math.pi) * x * z)
            sh_basis.append(0.25 * math.sqrt(15.0 / math.pi) * (x**2 - y**2))

        basis = torch.stack(sh_basis, dim=-1)

        num_basis = (deg + 1) ** 2
        if sh.shape[-1] != num_basis * 3:

            sh_coeffs = sh.view(*sh.shape[:-1], num_basis, 3)
        else:
            sh_coeffs = sh.view(*sh.shape[:-1], num_basis, 3)

        result = torch.sum(basis.unsqueeze(-1) * sh_coeffs, dim=-2)

        return result

class AirlightField(nn.Module):

    def __init__(self, mode='cnn', sh_degree=0, num_channels=3, feature_dim=64):
        super().__init__()
        self.mode = mode
        self.sh_degree = sh_degree
        self.num_channels = num_channels
        self.use_cnn = (mode == 'cnn')

        if self.use_cnn:

            self.degradation_extractor = DegradationFeatureExtractor(in_channels=3, feature_dim=feature_dim)
            self.airlight_cnn = LightweightCNN(feature_dim=feature_dim, output_channels=num_channels)

            self.global_airlight_bias = nn.Parameter(torch.randn(num_channels) * 1e-3)

        else:

            self.num_sh_coeffs = (sh_degree + 1) ** 2
            sh_coeffs = torch.zeros(self.num_sh_coeffs * num_channels)

            base_std_l0 = 0.01
            base_std_higher = 0.005

            if sh_degree == 0:

                sh_coeffs[:num_channels] = torch.randn(num_channels) * base_std_l0
                print(f"  0阶SH初始化: 均值0，小扰动N(0,{base_std_l0}) -> tanh映射初始≈0.5")
            else:

                sh_coeffs[:] = torch.randn_like(sh_coeffs) * base_std_higher
                print(f"  {sh_degree}阶SH初始化: 全系数N(0,{base_std_higher}) -> tanh映射初始≈0.5")

            self.sh_coeffs = nn.Parameter(sh_coeffs)

    def forward(self, positions, view_directions, clear_image=None, transmission=None):

        if self.use_cnn and clear_image is not None and transmission is not None:
            return self._predict_with_cnn(positions, view_directions, clear_image, transmission)
        else:
            return self._predict_with_sh(positions, view_directions)

    def _predict_with_cnn(self, positions, view_directions, clear_image, transmission):

        device = clear_image.device

        if clear_image.dim() == 3:
            clear_image_batch = clear_image.unsqueeze(0)
        else:
            clear_image_batch = clear_image

        if transmission.dim() == 2:
            transmission_batch = transmission.unsqueeze(0).unsqueeze(0)
        elif transmission.dim() == 3:
            transmission_batch = transmission.unsqueeze(0)
        else:
            transmission_batch = transmission

        degraded_features = self.degradation_extractor(clear_image_batch)

        airlight_map = self.airlight_cnn(degraded_features, transmission_batch)

        airlight_map = airlight_map + self.global_airlight_bias.view(1, self.num_channels, 1, 1)

        H, W = airlight_map.shape[2], airlight_map.shape[3]

        if positions.dim() == 2 and positions.shape[-1] == 3:

            N = positions.shape[0]

            pos_xy = positions[:, :2]

            pos_xy_norm = torch.clamp(pos_xy, -1, 1)

            sample_grid = pos_xy_norm.unsqueeze(0).unsqueeze(0)

            sampled_airlight = F.grid_sample(
                airlight_map, sample_grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )

            airlight = sampled_airlight.squeeze(0).squeeze(1).t()

        elif positions.dim() == 3:

            H_pos, W_pos = positions.shape[:2]

            if H_pos == H and W_pos == W:

                airlight = airlight_map.squeeze(0).permute(1, 2, 0)
            else:

                airlight_resized = F.interpolate(
                    airlight_map, size=(H_pos, W_pos),
                    mode='bilinear', align_corners=True
                )
                airlight = airlight_resized.squeeze(0).permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported position shape: {positions.shape}")

        return airlight

    def _predict_with_sh(self, positions, view_directions):

        view_directions = view_directions / (torch.norm(view_directions, dim=-1, keepdim=True) + 1e-8)

        airlight = SphericalHarmonics.eval_sh(
            self.sh_degree,
            self.sh_coeffs.unsqueeze(0).expand(*view_directions.shape[:-1], -1),
            view_directions
        )

        return airlight

    def get_regularization_loss(self):

        total_loss = 0.0

        if self.use_cnn:

            for module in [self.degradation_extractor, self.airlight_cnn]:
                for param in module.parameters():
                    total_loss += torch.sum(param**2) * 1e-6

            bias_loss = torch.sum(self.global_airlight_bias**2) * 1e-4
            total_loss += bias_loss

        else:

            if self.sh_degree > 0:
                high_freq_coeffs = self.sh_coeffs[3:]
                total_loss += torch.sum(high_freq_coeffs**2) * 1e-4

        return total_loss

    def get_high_freq_regularization(self):

        return self.get_regularization_loss()

    def eval_sh(self, directions):

        if self.use_cnn:

            result_shape = directions.shape[:-1] + (self.num_channels,)
            return self.global_airlight_bias.expand(result_shape)
        else:

            return self._predict_with_sh(None, directions)
