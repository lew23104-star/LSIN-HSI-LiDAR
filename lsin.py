"""Feature-level reference framework for Latent Scene Inference Network."""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Gaussian(NamedTuple):
    mean: Tensor
    log_variance: Tensor


class LatentFields(NamedTuple):
    joint: Gaussian
    hsi: Gaussian
    lidar: Gaussian


class ConditionalModulation(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.source_norm = nn.LayerNorm(width)
        self.receiver_norm = nn.LayerNorm(width)
        self.conditioner = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width)
        )
        self.update = nn.Linear(width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.conditioner[-1].weight, std=0.02)
        nn.init.zeros_(self.conditioner[-1].bias)
        nn.init.normal_(self.update.weight, std=0.02)

    def forward(self, receiver: Tensor, source: Tensor) -> Tensor:
        scale, shift = self.conditioner(self.source_norm(source)).chunk(2, dim=-1)
        modulated = (1 + scale.tanh()) * self.receiver_norm(receiver) + shift.tanh()
        return self.output_norm(receiver + self.update(F.gelu(modulated)))


def normalised_precision_pool(native: Gaussian, conditional: Gaussian) -> tuple[Tensor, Tensor]:
    """Return the group's precision and precision-weighted mean."""
    native_precision = (-native.log_variance).exp()
    conditional_precision = (-conditional.log_variance).exp()
    native_weight = native_precision / (native_precision + conditional_precision)
    native_share = native_weight * native_precision
    conditional_share = (1 - native_weight) * conditional_precision
    return (
        native_share + conditional_share,
        native_share * native.mean + conditional_share * conditional.mean,
    )


def single_sensor_field(factor: Gaussian) -> Gaussian:
    precision = (-factor.log_variance).exp()
    posterior_precision = 1 + precision
    return Gaussian(precision * factor.mean / posterior_precision, -posterior_precision.log())


class LSIN(nn.Module):
    """Fuse aligned native HSI and LiDAR feature tokens of shape [B, N, D]."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        patch_size: int,
        latent_dim: int = 96,
        refiner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if latent_dim < 3 or latent_dim % 3:
            raise ValueError("latent_dim must be a positive multiple of three")
        self.feature_dim = feature_dim
        self.num_positions = patch_size**2
        self.centre = self.num_positions // 2
        self.lidar_to_hsi = ConditionalModulation(feature_dim)
        self.hsi_to_lidar = ConditionalModulation(feature_dim)
        self.gaussian_mapping = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 2 * latent_dim)
        )
        self.refiner = refiner if refiner is not None else nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * latent_dim), nn.Linear(2 * latent_dim, num_classes)
        )

    def _gaussian(self, view: Tensor) -> Gaussian:
        mean, raw = self.gaussian_mapping(view).chunk(2, dim=-1)
        return Gaussian(mean, -6 + 10 * raw.sigmoid())

    def latent_fields(self, hsi: Tensor, lidar: Tensor) -> LatentFields:
        expected = (self.num_positions, self.feature_dim)
        if hsi.ndim != 3 or hsi.shape[1:] != expected or lidar.shape != hsi.shape:
            raise ValueError(f"Expected two aligned inputs of shape [B, {expected[0]}, {expected[1]}]")

        hsi_native, lidar_native = self._gaussian(hsi), self._gaussian(lidar)
        hsi_conditional = self._gaussian(self.lidar_to_hsi(receiver=hsi, source=lidar))
        lidar_conditional = self._gaussian(self.hsi_to_lidar(receiver=lidar, source=hsi))
        hsi_precision, hsi_natural_mean = normalised_precision_pool(hsi_native, hsi_conditional)
        lidar_precision, lidar_natural_mean = normalised_precision_pool(lidar_native, lidar_conditional)
        joint_precision = 1 + hsi_precision + lidar_precision
        joint = Gaussian(
            (hsi_natural_mean + lidar_natural_mean) / joint_precision, -joint_precision.log()
        )
        return LatentFields(joint, single_sensor_field(hsi_native), single_sensor_field(lidar_native))

    def classify(self, field: Tensor) -> Tensor:
        refined = self.refiner(field)
        return self.classifier(torch.cat((refined[:, self.centre], refined.mean(dim=1)), dim=-1))

    def forward(self, hsi: Tensor, lidar: Tensor) -> Tensor:
        return self.classify(self.latent_fields(hsi, lidar).joint.mean)


class ObservationDecoders(nn.Module):
    """Share user-supplied target decoders across reconstruction and prediction."""

    def __init__(self, hsi_centre: nn.Module, hsi_relative: nn.Module, lidar: nn.Module) -> None:
        super().__init__()
        self.hsi_centre = hsi_centre
        self.hsi_relative = hsi_relative
        self.lidar = lidar

    def _decode_hsi(self, field: Tensor) -> tuple[Tensor, Tensor]:
        material, _, shared = field.chunk(3, dim=-1)
        spectral = torch.cat((material, shared), dim=-1)
        return self.hsi_centre(spectral[:, field.shape[1] // 2]), self.hsi_relative(spectral)

    def _decode_lidar(self, field: Tensor) -> Tensor:
        _, geometry, shared = field.chunk(3, dim=-1)
        return self.lidar(torch.cat((geometry, shared), dim=-1))

    def forward(self, fields: LatentFields) -> dict[str, Tensor]:
        centre, relative = self._decode_hsi(fields.joint.mean)
        centre_from_lidar, relative_from_lidar = self._decode_hsi(fields.lidar.mean)
        return {
            "joint_hsi_centre": centre,
            "joint_hsi_relative": relative,
            "joint_lidar": self._decode_lidar(fields.joint.mean),
            "hsi_centre_from_lidar": centre_from_lidar,
            "hsi_relative_from_lidar": relative_from_lidar,
            "lidar_from_hsi": self._decode_lidar(fields.hsi.mean),
        }
