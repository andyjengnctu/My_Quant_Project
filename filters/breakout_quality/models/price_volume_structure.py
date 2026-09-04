"""PIT-safe price/volume structural representation for Safety learning.

The primitive consumes the existing canonical normalized stock OHLCV channels
from the 300-bar sequence.  It does not add a second data source, technical
indicator truth, hand-labelled support/resistance level, or persisted expanded
feature bank.  A parameter-free rasterizer exposes price/time geometry and
relative-volume-at-price evidence; small trainable CNN branches encode those
representations for Safety-only residual fusion.
"""

from __future__ import annotations

import math


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

        def encode_structure_inputs(self, geometry_map, vap):
            geometry_latent = self.geometry_network(geometry_map).flatten(1)
            vap_latent = self.vap_network(vap.unsqueeze(1)).flatten(1)
            return self.fusion_projection(
                torch.cat((geometry_latent, vap_latent), dim=1)
            )

        def forward(self, x):
            geometry_map, vap = self.build_structure_inputs(x)
            return self.encode_structure_inputs(geometry_map, vap)

    return PriceVolumeStructureEncoder()


def build_price_volume_local_structure_encoder(
    nn,
    torch,
    *,
    output_width: int,
    spec,
):
    """Encode nearby price-zone evidence as query-conditioned structure tokens.

    The dense BL field remains intact.  This additive primitive reuses the exact
    same PIT raster and treats nearby price bins as unordered zone tokens.  A
    single-head query from the globally conditioned Safety latent decides which
    zones matter for the current sample; no pivot labels, support/resistance
    labels, top-K zone selection, or hand-weighted strength score is introduced.
    """

    price_bins = int(spec.price_volume_structure_price_bins or 0)
    price_span_atr = float(spec.price_volume_structure_price_span_atr or 0.0)
    local_span_atr = float(spec.price_volume_local_structure_span_atr or 0.0)
    token_dim = int(spec.price_volume_local_structure_token_dim or 0)
    recent_fraction = float(spec.price_volume_local_structure_recent_fraction or 0.0)
    if price_bins < 8 or price_span_atr <= 0.0:
        raise ValueError("Local Price-Volume structure需要合法的global price-grid contract")
    if not (0.0 < local_span_atr < price_span_atr):
        raise ValueError("Local Price-Volume span必須介於0與global span之間")
    if token_dim < 1 or int(output_width) < 1:
        raise ValueError("Local Price-Volume token/output width必須為正")
    if not (0.0 < recent_fraction <= 1.0):
        raise ValueError("Local Price-Volume recent fraction必須位於(0,1]")

    token_feature_dim = 11
    attention_scale = float(token_dim) ** -0.5

    class PriceVolumeLocalStructureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_projection = nn.Linear(token_feature_dim, token_dim, bias=False)
            self.query_projection = nn.Linear(int(output_width), token_dim, bias=False)
            self.key_projection = nn.Linear(token_dim, token_dim, bias=False)
            self.value_projection = nn.Linear(token_dim, token_dim, bias=False)
            self.output_projection = nn.Linear(token_dim, int(output_width), bias=False)

        @staticmethod
        def _price_axis(device):
            return torch.linspace(
                -price_span_atr,
                price_span_atr,
                steps=price_bins,
                device=device,
                dtype=torch.float32,
            )

        def build_local_tokens(self, geometry_map, vap):
            if geometry_map.ndim != 4 or int(geometry_map.shape[1]) != 4:
                raise ValueError("Local Price-Volume geometry必須是[B,4,T,P]")
            if int(geometry_map.shape[3]) != price_bins:
                raise ValueError("Local Price-Volume price bins與model spec不一致")
            if vap.ndim != 2 or int(vap.shape[1]) != price_bins:
                raise ValueError("Local Price-Volume VAP shape與model spec不一致")

            with torch.no_grad():
                axis = self._price_axis(geometry_map.device)
                local_mask = torch.abs(axis) <= local_span_atr
                if int(local_mask.sum().item()) < 2:
                    raise ValueError("Local Price-Volume span至少必須涵蓋2個price bins")

                local_map = geometry_map[:, :, :, local_mask].float()
                local_vap = vap[:, local_mask].float()
                local_axis = axis[local_mask]
                time_count = int(local_map.shape[2])
                recent_count = max(1, int(math.ceil(time_count * recent_fraction)))

                full_mean = local_map.mean(dim=2)
                recent_mean = local_map[:, :, -recent_count:, :].mean(dim=2)

                # Volume-weighted structural recency: 0=old edge, 1=latest edge.
                volume_mass = local_map[:, 2:4, :, :].sum(dim=1)
                time_axis = torch.linspace(
                    0.0,
                    1.0,
                    steps=time_count,
                    device=geometry_map.device,
                    dtype=torch.float32,
                )[None, :, None]
                mass_sum = volume_mass.sum(dim=1)
                eps = torch.finfo(volume_mass.dtype).eps
                recency = (volume_mass * time_axis).sum(dim=1) / mass_sum.clamp_min(eps)
                recency = torch.where(mass_sum > 0.0, recency, torch.zeros_like(recency))

                signed_distance = (
                    local_axis / local_span_atr
                )[None, :].expand(int(local_map.shape[0]), -1)

                tokens = torch.stack(
                    (
                        signed_distance,
                        full_mean[:, 0, :],
                        full_mean[:, 1, :],
                        full_mean[:, 2, :],
                        full_mean[:, 3, :],
                        recent_mean[:, 0, :],
                        recent_mean[:, 1, :],
                        recent_mean[:, 2, :],
                        recent_mean[:, 3, :],
                        local_vap,
                        recency,
                    ),
                    dim=2,
                )
                evidence_mass = (
                    full_mean[:, 0, :]
                    + full_mean[:, 1, :]
                    + full_mean[:, 2, :]
                    + full_mean[:, 3, :]
                )
                evidence_mask = evidence_mass > 0.0

            return tokens, evidence_mask

        def attention_weights(self, geometry_map, vap, query_source):
            tokens, evidence_mask = self.build_local_tokens(geometry_map, vap)
            token_latent = torch.relu(self.token_projection(tokens))
            token_latent = token_latent * evidence_mask.unsqueeze(2).to(token_latent.dtype)
            query = self.query_projection(query_source).unsqueeze(1)
            keys = self.key_projection(token_latent)
            scores = torch.sum(query * keys, dim=2) * attention_scale
            scores = scores.masked_fill(~evidence_mask, -1.0e4)
            return torch.softmax(scores.float(), dim=1).to(token_latent.dtype)

        def forward(self, geometry_map, vap, query_source):
            tokens, evidence_mask = self.build_local_tokens(geometry_map, vap)
            token_latent = torch.relu(self.token_projection(tokens))
            token_latent = token_latent * evidence_mask.unsqueeze(2).to(token_latent.dtype)
            query = self.query_projection(query_source).unsqueeze(1)
            keys = self.key_projection(token_latent)
            values = self.value_projection(token_latent)
            scores = torch.sum(query * keys, dim=2) * attention_scale
            scores = scores.masked_fill(~evidence_mask, -1.0e4)
            weights = torch.softmax(scores.float(), dim=1).to(values.dtype)
            context = torch.sum(values * weights.unsqueeze(2), dim=1)
            return self.output_projection(context)

    return PriceVolumeLocalStructureEncoder()


def build_price_volume_multiscale_local_encoder(
    nn,
    torch,
    *,
    feature_count: int,
    output_width: int,
    spec,
):
    """High-resolution recent Price×Volume geometry with explicit global interaction.

    The BL global field is preserved unchanged.  This branch independently
    rasterizes only the most recent bars at a tighter ATR span, keeping the
    time×price geometry that zone-token aggregation discards.  Its residual
    combines a local 2D latent with an explicit element-wise interaction against
    the already-computed BL global residual.  It never feeds the MFE head.
    """

    if int(feature_count) < 5:
        raise ValueError("Multi-scale local Price-Volume需要stock OHLCV前5個sequence channels")
    history_bars = int(spec.price_volume_local_map_history_bars or 0)
    time_bins = int(spec.price_volume_local_map_time_bins or 0)
    price_bins = int(spec.price_volume_local_map_price_bins or 0)
    price_span_atr = float(spec.price_volume_local_map_price_span_atr or 0.0)
    latent_dim = int(spec.price_volume_local_map_geometry_latent_dim or 0)
    atr_bars = int(spec.price_volume_structure_atr_bars or 0)
    geometry_channels = int(spec.price_volume_structure_geometry_channels or 0)
    if history_bars < atr_bars or history_bars < 8:
        raise ValueError("Multi-scale local Price-Volume history必須涵蓋ATR且>=8 bars")
    if time_bins < 8 or price_bins < 8 or price_span_atr <= 0.0:
        raise ValueError("Multi-scale local Price-Volume raster contract不合法")
    if geometry_channels != 4 or latent_dim < 1 or int(output_width) < 1:
        raise ValueError("Multi-scale local Price-Volume channels/latent/output contract不合法")

    functional = torch.nn.functional

    class PriceVolumeMultiscaleLocalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_geometry_network = nn.Sequential(
                nn.Conv2d(geometry_channels, 12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, latent_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            # Explicit cross-scale interaction: local geometry is retained as a
            # main effect and multiplied by a projected global structural field.
            self.global_projection = nn.Linear(int(output_width), latent_dim, bias=False)
            self.fusion_projection = nn.Linear(2 * latent_dim, int(output_width), bias=False)

        @staticmethod
        def _relative_volume_weight(volume_z):
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
            eps = torch.finfo(true_range.dtype).eps
            return true_range[:, -width:].mean(dim=1).clamp_min(eps)

        def build_local_geometry_map(self, x):
            if x.ndim != 3 or int(x.shape[2]) < 5:
                raise ValueError("Multi-scale local Price-Volume input必須是[B,T,F>=5]")
            if int(x.shape[1]) < history_bars:
                raise ValueError("Multi-scale local Price-Volume input history不足")
            with torch.no_grad():
                stock = x[:, -history_bars:, :5].float()
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
                half_bin = 0.5 * current_atr[:, None, None] * (
                    2.0 * price_span_atr / float(price_bins - 1)
                )
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
                full_map = torch.stack(
                    (body, wick, body * relative_volume, wick * relative_volume),
                    dim=1,
                )
                # 80 observed bars -> 96 deterministic display/conv bins.  Linear
                # interpolation preserves local diagonal/horizontal geometry without
                # inventing another market-data source or learnable preprocessing.
                local_map = functional.interpolate(
                    full_map,
                    size=(time_bins, price_bins),
                    mode="bilinear",
                    align_corners=False,
                )
            return local_map

        def encode_local_geometry_map(self, local_map, global_residual):
            local_latent = self.local_geometry_network(local_map).flatten(1)
            # No-evidence local windows must not manufacture a learned constant
            # structure solely from Conv biases.
            evidence_mask = (
                local_map.detach().abs().sum(dim=(1, 2, 3), keepdim=False) > 0.0
            ).to(local_latent.dtype).unsqueeze(1)
            local_latent = local_latent * evidence_mask
            global_latent = self.global_projection(global_residual)
            interaction = local_latent * global_latent
            return self.fusion_projection(torch.cat((local_latent, interaction), dim=1))

        def forward(self, x, global_residual):
            local_map = self.build_local_geometry_map(x)
            return self.encode_local_geometry_map(local_map, global_residual)

    return PriceVolumeMultiscaleLocalEncoder()


def build_price_volume_position_aware_multiscale_encoder(
    nn,
    torch,
    *,
    feature_count: int,
    output_width: int,
    spec,
):
    """Position-aware global/local Price×Volume geometry for Safety-only fusion.

    MR-13BO keeps the BN raster spans/resolutions and AO dual-head scientific
    contract, but explicitly exposes absolute geometry coordinates to the CNNs.
    Evidence channels remain body/wick/relative-volume×body/relative-volume×wick.
    Two deterministic CoordConv-style channels are appended: signed price
    position relative to the decision-day close (normalized by each branch ATR
    span) and time age from oldest=-1 to newest=+1.  VAP similarly receives the
    signed price coordinate as a second channel.  No coordinate reaches MFE.
    """

    if int(feature_count) < 5:
        raise ValueError("Position-aware Price-Volume需要stock OHLCV前5個sequence channels")
    global_time_bins = int(spec.price_volume_structure_time_bins or 0)
    global_price_bins = int(spec.price_volume_structure_price_bins or 0)
    global_span_atr = float(spec.price_volume_structure_price_span_atr or 0.0)
    atr_bars = int(spec.price_volume_structure_atr_bars or 0)
    geometry_channels = int(spec.price_volume_structure_geometry_channels or 0)
    geometry_latent_dim = int(spec.price_volume_structure_geometry_latent_dim or 0)
    vap_channels = int(spec.price_volume_structure_vap_channels or 0)
    vap_latent_dim = int(spec.price_volume_structure_vap_latent_dim or 0)
    coordinate_mode = str(spec.price_volume_structure_coordinate_mode or "")
    local_history_bars = int(spec.price_volume_local_map_history_bars or 0)
    local_time_bins = int(spec.price_volume_local_map_time_bins or 0)
    local_price_bins = int(spec.price_volume_local_map_price_bins or 0)
    local_span_atr = float(spec.price_volume_local_map_price_span_atr or 0.0)
    local_latent_dim = int(spec.price_volume_local_map_geometry_latent_dim or 0)

    if global_time_bins < 8 or global_price_bins < 8 or global_span_atr <= 0.0:
        raise ValueError("Position-aware global Price-Volume raster contract不合法")
    if local_history_bars < atr_bars or local_history_bars < 8:
        raise ValueError("Position-aware local Price-Volume history必須涵蓋ATR且>=8 bars")
    if local_time_bins < 8 or local_price_bins < 8 or local_span_atr <= 0.0:
        raise ValueError("Position-aware local Price-Volume raster contract不合法")
    if geometry_channels != 6:
        raise ValueError("Position-aware Price-Volume固定6 channels: 4 evidence + signed-price + time-age")
    if vap_channels != 2:
        raise ValueError("Position-aware VAP固定2 channels: mass + signed-price")
    if coordinate_mode != "signed_atr_price_plus_time_age":
        raise ValueError("Position-aware Price-Volume coordinate mode不一致")
    if geometry_latent_dim < 1 or vap_latent_dim < 1 or local_latent_dim < 1 or int(output_width) < 1:
        raise ValueError("Position-aware Price-Volume latent/output width必須為正")

    functional = torch.nn.functional

    class PriceVolumePositionAwareMultiscaleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_geometry_network = nn.Sequential(
                nn.Conv2d(geometry_channels, 12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, geometry_latent_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.vap_network = nn.Sequential(
                nn.Conv1d(vap_channels, 8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(8, vap_latent_dim, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.global_fusion_projection = nn.Linear(
                geometry_latent_dim + vap_latent_dim,
                int(output_width),
                bias=False,
            )
            self.local_geometry_network = nn.Sequential(
                nn.Conv2d(geometry_channels, 12, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, local_latent_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.global_projection = nn.Linear(int(output_width), local_latent_dim, bias=False)
            self.local_fusion_projection = nn.Linear(
                2 * local_latent_dim,
                int(output_width),
                bias=False,
            )

        @staticmethod
        def _relative_volume_weight(volume_z):
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
            eps = torch.finfo(true_range.dtype).eps
            return true_range[:, -width:].mean(dim=1).clamp_min(eps)

        @staticmethod
        def _append_coordinates(evidence_map):
            batch, _channels, time_count, price_count = evidence_map.shape
            price_coordinate = torch.linspace(
                -1.0, 1.0, steps=int(price_count), device=evidence_map.device, dtype=torch.float32
            ).view(1, 1, 1, int(price_count)).expand(int(batch), 1, int(time_count), int(price_count))
            time_coordinate = torch.linspace(
                -1.0, 1.0, steps=int(time_count), device=evidence_map.device, dtype=torch.float32
            ).view(1, 1, int(time_count), 1).expand(int(batch), 1, int(time_count), int(price_count))
            return torch.cat((evidence_map, price_coordinate, time_coordinate), dim=1)

        @staticmethod
        def _position_aware_vap(vap):
            batch, price_count = vap.shape
            price_coordinate = torch.linspace(
                -1.0, 1.0, steps=int(price_count), device=vap.device, dtype=torch.float32
            ).view(1, 1, int(price_count)).expand(int(batch), 1, int(price_count))
            return torch.cat((vap.unsqueeze(1), price_coordinate), dim=1)

        def _raster_evidence(self, stock, *, price_bins, price_span_atr):
            open_price = stock[:, :, 0]
            high = stock[:, :, 1]
            low = stock[:, :, 2]
            close = stock[:, :, 3]
            volume_z = stock[:, :, 4]
            current_atr = self._current_atr_return(high, low, close)
            atr_axis = torch.linspace(
                -float(price_span_atr),
                float(price_span_atr),
                steps=int(price_bins),
                device=stock.device,
                dtype=torch.float32,
            )
            # Canonical stock price channels are already normalized to the
            # decision-day close, so zero is the current-price anchor.
            price_centers = current_atr[:, None, None] * atr_axis[None, None, :]
            half_bin = 0.5 * current_atr[:, None, None] * (
                2.0 * float(price_span_atr) / float(int(price_bins) - 1)
            )
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
            return torch.stack(
                (body, wick, body * relative_volume, wick * relative_volume),
                dim=1,
            )

        def build_global_position_aware_inputs(self, x):
            if x.ndim != 3 or int(x.shape[2]) < 5:
                raise ValueError("Position-aware global Price-Volume input必須是[B,T,F>=5]")
            if int(x.shape[1]) != int(spec.input_window_bars or x.shape[1]):
                raise ValueError("Position-aware global Price-Volume sequence length與model spec不一致")
            with torch.no_grad():
                stock = x[:, :, :5].float()
                full_evidence = self._raster_evidence(
                    stock,
                    price_bins=global_price_bins,
                    price_span_atr=global_span_atr,
                )
                evidence_map = functional.adaptive_avg_pool2d(
                    full_evidence,
                    output_size=(global_time_bins, global_price_bins),
                )
                position_aware_map = self._append_coordinates(evidence_map)
                vap = (full_evidence[:, 2] + full_evidence[:, 3]).sum(dim=1)
                vap_mean = vap.mean(dim=1, keepdim=True)
                eps = torch.finfo(vap.dtype).eps
                vap = vap / vap_mean.clamp_min(eps)
                position_aware_vap = self._position_aware_vap(vap)
            return position_aware_map, position_aware_vap

        def build_local_position_aware_map(self, x):
            if x.ndim != 3 or int(x.shape[2]) < 5:
                raise ValueError("Position-aware local Price-Volume input必須是[B,T,F>=5]")
            if int(x.shape[1]) < local_history_bars:
                raise ValueError("Position-aware local Price-Volume input history不足")
            with torch.no_grad():
                stock = x[:, -local_history_bars:, :5].float()
                full_evidence = self._raster_evidence(
                    stock,
                    price_bins=local_price_bins,
                    price_span_atr=local_span_atr,
                )
                evidence_map = functional.interpolate(
                    full_evidence,
                    size=(local_time_bins, local_price_bins),
                    mode="bilinear",
                    align_corners=False,
                )
                position_aware_map = self._append_coordinates(evidence_map)
            return position_aware_map

        def encode_global(self, global_map, position_aware_vap):
            global_latent = self.global_geometry_network(global_map).flatten(1)
            vap_latent = self.vap_network(position_aware_vap).flatten(1)
            return self.global_fusion_projection(torch.cat((global_latent, vap_latent), dim=1))

        def encode_local(self, local_map, global_residual):
            local_latent = self.local_geometry_network(local_map).flatten(1)
            # Coordinate channels are non-zero by construction. Evidence gating
            # must therefore inspect only the four market-evidence channels.
            evidence_mask = (
                local_map[:, :4].detach().abs().sum(dim=(1, 2, 3), keepdim=False) > 0.0
            ).to(local_latent.dtype).unsqueeze(1)
            local_latent = local_latent * evidence_mask
            global_latent = self.global_projection(global_residual)
            interaction = local_latent * global_latent
            return self.local_fusion_projection(torch.cat((local_latent, interaction), dim=1))

        def forward(self, x):
            global_map, position_aware_vap = self.build_global_position_aware_inputs(x)
            global_residual = self.encode_global(global_map, position_aware_vap)
            local_map = self.build_local_position_aware_map(x)
            local_residual = self.encode_local(local_map, global_residual)
            return global_residual + local_residual

    return PriceVolumePositionAwareMultiscaleEncoder()


__all__ = [
    "build_price_volume_structure_encoder",
    "build_price_volume_local_structure_encoder",
    "build_price_volume_multiscale_local_encoder",
    "build_price_volume_position_aware_multiscale_encoder",
]
