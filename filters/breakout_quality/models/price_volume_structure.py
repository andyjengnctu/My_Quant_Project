"""PIT-safe price/volume structural representation for Safety learning.

The primitive consumes the existing canonical normalized stock OHLCV channels
from the 300-bar sequence.  It does not add a second data source, technical
indicator truth, hand-labelled support/resistance level, or persisted expanded
feature bank.  A parameter-free rasterizer exposes price/time geometry and
relative-volume-at-price evidence; small trainable CNN branches encode those
representations for Safety-only residual fusion.
"""

from __future__ import annotations


def build_price_volume_structure_encoder(
    nn,
    torch,
    *,
    feature_count: int,
    output_width: int,
    spec,
):
    if int(feature_count) < 5:
        raise ValueError("Price-Volume structure需要stock OHLCV前5個sequence channels")

    time_bins = int(spec.price_volume_structure_time_bins or 0)
    price_bins = int(spec.price_volume_structure_price_bins or 0)
    price_span_atr = float(spec.price_volume_structure_price_span_atr or 0.0)
    atr_bars = int(spec.price_volume_structure_atr_bars or 0)
    geometry_channels = int(spec.price_volume_structure_geometry_channels or 0)
    geometry_latent_dim = int(spec.price_volume_structure_geometry_latent_dim or 0)
    vap_latent_dim = int(spec.price_volume_structure_vap_latent_dim or 0)
    if time_bins < 8 or price_bins < 8:
        raise ValueError("Price-Volume structure time/price bins必須 >= 8")
    if price_span_atr <= 0.0 or atr_bars < 2:
        raise ValueError("Price-Volume structure ATR span/bars必須為正且atr_bars>=2")
    if geometry_channels != 4:
        raise ValueError("Price-Volume structure v1固定4 channels: body/wick/volume-body/volume-wick")
    if geometry_latent_dim < 1 or vap_latent_dim < 1 or int(output_width) < 1:
        raise ValueError("Price-Volume structure latent/output width必須為正")

    functional = torch.nn.functional

    class PriceVolumeStructureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.geometry_network = nn.Sequential(
                nn.Conv2d(geometry_channels, 12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, geometry_latent_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.vap_network = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(8, vap_latent_dim, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            # Bias-free residual projection prevents a learned constant Safety shift.
            self.fusion_projection = nn.Linear(
                geometry_latent_dim + vap_latent_dim,
                int(output_width),
                bias=False,
            )

        @staticmethod
        def _relative_volume_weight(volume_z):
            """Positive monotone relative-volume mass with mean weight exactly one.

            The canonical volume input is window-normalized log-volume. Softplus
            keeps the transformation parameter-free and monotone; mean normalization
            makes the structural signal represent within-window relative activity
            instead of ticker scale.
            """

            positive = functional.softplus(volume_z.float())
            mean = positive.mean(dim=1, keepdim=True)
            eps = torch.finfo(positive.dtype).eps
            return positive / mean.clamp_min(eps)

        @staticmethod
        def _current_atr_return(high, low, close):
            previous_close = torch.cat((close[:, :1], close[:, :-1]), dim=1)
            true_range = torch.maximum(
                high - low,
                torch.maximum(
                    torch.abs(high - previous_close),
                    torch.abs(low - previous_close),
                ),
            )
            width = min(int(atr_bars), int(true_range.shape[1]))
            current_atr = true_range[:, -width:].mean(dim=1)
            eps = torch.finfo(true_range.dtype).eps
            return current_atr.clamp_min(eps)

        def build_structure_inputs(self, x):
            """Return deterministic dense geometry map and normalized VAP profile.

            Output shapes are ``[B,4,time_bins,price_bins]`` and
            ``[B,price_bins]``.  Rasterization is non-trainable input preparation;
            trainable gradients begin at the CNN encoders.
            """

            if x.ndim != 3 or int(x.shape[2]) < 5:
                raise ValueError(
                    "Price-Volume structure input必須是 [batch,time,feature>=5]"
                )
            if int(x.shape[1]) != int(spec.input_window_bars or x.shape[1]):
                raise ValueError("Price-Volume structure sequence length與model spec不一致")

            # Use FP32 for bin geometry even when the surrounding model is under
            # BF16 autocast.  No input gradient is required for deterministic
            # representation construction.
            with torch.no_grad():
                stock = x[:, :, :5].float()
                open_price = stock[:, :, 0]
                high = stock[:, :, 1]
                low = stock[:, :, 2]
                close = stock[:, :, 3]
                volume_z = stock[:, :, 4]

                current_atr = self._current_atr_return(high, low, close)
                atr_axis = torch.linspace(
                    -price_span_atr,
                    price_span_atr,
                    steps=price_bins,
                    device=x.device,
                    dtype=torch.float32,
                )
                price_centers = current_atr[:, None, None] * atr_axis[None, None, :]
                atr_bin_width = (
                    current_atr[:, None, None]
                    * (2.0 * price_span_atr / float(price_bins - 1))
                )
                half_bin = 0.5 * atr_bin_width

                body_low = torch.minimum(open_price, close)[:, :, None]
                body_high = torch.maximum(open_price, close)[:, :, None]
                candle_low = low[:, :, None]
                candle_high = high[:, :, None]

                body = (
                    (price_centers >= body_low - half_bin)
                    & (price_centers <= body_high + half_bin)
                ).float()
                full_range = (
                    (price_centers >= candle_low - half_bin)
                    & (price_centers <= candle_high + half_bin)
                ).float()
                wick = torch.clamp(full_range - body, min=0.0, max=1.0)

                relative_volume = self._relative_volume_weight(volume_z)[:, :, None]
                volume_body = body * relative_volume
                volume_wick = wick * relative_volume

                full_map = torch.stack(
                    (body, wick, volume_body, volume_wick), dim=1
                )
                geometry_map = functional.adaptive_avg_pool2d(
                    full_map,
                    output_size=(time_bins, price_bins),
                )

                vap = (volume_body + volume_wick).sum(dim=1)
                vap_mean = vap.mean(dim=1, keepdim=True)
                eps = torch.finfo(vap.dtype).eps
                vap = vap / vap_mean.clamp_min(eps)

            return geometry_map, vap

        def forward(self, x):
            geometry_map, vap = self.build_structure_inputs(x)
            geometry_latent = self.geometry_network(geometry_map).flatten(1)
            vap_latent = self.vap_network(vap.unsqueeze(1)).flatten(1)
            return self.fusion_projection(
                torch.cat((geometry_latent, vap_latent), dim=1)
            )

    return PriceVolumeStructureEncoder()


__all__ = ["build_price_volume_structure_encoder"]
