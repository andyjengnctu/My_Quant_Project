"""InceptionTime daily encoder with small masked calendar-sequence branches."""

from __future__ import annotations

from filters.breakout_quality.calendar_bars import (
    CALENDAR_CHANNELS, DAILY_CHANNELS, calendar_bar_layout,
)
from filters.breakout_quality.models.inception_time import build_inception_time


def build_calendar_inception(nn, torch, *, feature_count: int, context_count: int, spec):
    layout = calendar_bar_layout(spec.calendar_bar_windows)
    if feature_count != DAILY_CHANNELS + len(layout) * CALENDAR_CHANNELS:
        raise ValueError("Packed calendar model feature count differs from its contract")
    channels = int(spec.calendar_encoder_channels)
    depth = int(spec.calendar_encoder_depth)
    kernel = int(spec.calendar_encoder_kernel_size)
    fusion_width = int(spec.calendar_fusion_width)
    if min(channels, depth, kernel, fusion_width) < 1 or kernel % 2 != 1:
        raise ValueError("Calendar encoder widths/depth must be positive and kernel odd")

    class MaskedCalendarEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.convolutions = nn.ModuleList([
                nn.Conv1d(CALENDAR_CHANNELS if index == 0 else channels, channels, kernel, padding=kernel // 2)
                for index in range(depth)
            ])

        def forward(self, x):
            stock_valid, benchmark_valid = x[:, :, 5:6], x[:, :, 11:12]
            observed = torch.cat((
                x[:, :, :5] * stock_valid, stock_valid,
                x[:, :, 6:11] * benchmark_valid, benchmark_valid,
            ), dim=2)
            mask = torch.maximum(stock_valid, benchmark_valid).transpose(1, 2)
            z = observed.transpose(1, 2)
            for convolution in self.convolutions:
                z = torch.relu(convolution(z)) * mask
            return z.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    class CalendarInceptionRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.daily_encoder = build_inception_time(
                nn, torch, feature_count=DAILY_CHANNELS, context_count=context_count, spec=spec,
            )
            daily_width = self.daily_encoder.classifier.in_features
            self.daily_encoder.classifier = nn.Identity()
            self.calendar_encoders = nn.ModuleList([MaskedCalendarEncoder() for _ in layout])
            self.fusion = nn.Sequential(nn.Linear(daily_width + len(layout) * channels, fusion_width), nn.ReLU())
            self.default_output_width = int(spec.classifier_output_classes or 2)
            self.classifier = nn.Linear(fusion_width, self.default_output_width)

        def encode(self, x):
            if x.shape[1] != spec.input_window_bars or x.shape[2] != feature_count:
                raise ValueError("Packed calendar input shape differs from its model contract")
            latents = [self.daily_encoder.encode(x[:, :, :DAILY_CHANNELS])]
            for item, encoder in zip(layout, self.calendar_encoders, strict=True):
                start = item["channel_start"]
                latents.append(encoder(x[:, -item["length"]:, start:start + CALENDAR_CHANNELS]))
            return self.fusion(torch.cat(latents, dim=1))

        def forward(self, x, context, *, relation_history_x=None):
            return self.classifier(self.encode(x))

        def forward_output_head(self, x, context, output_head, *, relation_history_x=None):
            if str(output_head).strip().lower() not in {"primary", "mfe", "primary_mfe"}:
                raise ValueError(f"Unsupported single-ranker output head: {output_head!r}")
            return self.forward(x, context, relation_history_x=relation_history_x)

    return CalendarInceptionRanker()
