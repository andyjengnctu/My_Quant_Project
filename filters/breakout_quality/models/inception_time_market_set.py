"""InceptionTime candidate branch plus a learned full-market set encoder."""

from __future__ import annotations

from filters.breakout_quality.models.inception_time import build_inception_time


def build_inception_time_market_set(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
):
    if bool(spec.use_dataset_context):
        raise ValueError("Market Set architecture 必須停用 Dataset event context")
    if not bool(spec.requires_market_set):
        raise ValueError("Market Set architecture spec 必須 requires_market_set=true")

    stock_embedding_dim = int(spec.market_set_stock_embedding_dim)
    temporal_channels = int(spec.market_set_temporal_channels)
    temporal_kernel = int(spec.market_set_temporal_kernel_size)
    temporal_stride = int(spec.market_set_temporal_stride)
    temporal_dilations = tuple(int(value) for value in spec.market_set_temporal_dilations)
    query_count = int(spec.market_set_query_count)
    attention_heads = int(spec.market_set_attention_heads)
    market_embedding_dim = int(spec.market_set_embedding_dim)
    fusion_hidden_dim = int(spec.market_set_fusion_hidden_dim)
    market_feature_count = len(tuple(spec.market_set_base_features)) + 1  # explicit history mask
    temporal_normalization = str(spec.market_set_temporal_normalization)
    normalization_groups = int(spec.market_set_temporal_normalization_groups)

    if stock_embedding_dim < 1 or temporal_channels < 1:
        raise ValueError("market stock embedding／temporal channels 必須 >=1")
    if temporal_kernel < 1 or temporal_stride < 1:
        raise ValueError("market temporal kernel／stride 必須 >=1")
    if not temporal_dilations or any(value < 1 for value in temporal_dilations):
        raise ValueError("market temporal dilations 必須是非空正整數")
    if query_count < 1 or attention_heads < 1:
        raise ValueError("market query_count／attention_heads 必須 >=1")
    if stock_embedding_dim % attention_heads != 0:
        raise ValueError("market stock embedding 必須可被 attention heads 整除")
    if temporal_normalization != "group_norm":
        raise ValueError("market temporal normalization 第一版固定為 group_norm")
    if normalization_groups < 1 or temporal_channels % normalization_groups != 0:
        raise ValueError("market temporal normalization groups 不合法")

    class ResidualTemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            padding = dilation
            self.network = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    dilation=dilation,
                    padding=padding,
                    groups=channels,
                    bias=False,
                ),
                nn.GroupNorm(normalization_groups, channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(normalization_groups, channels),
            )
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(x + self.network(x))

    class SharedStockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_kernel = temporal_kernel
            self.temporal_stride = temporal_stride
            self.temporal_padding = temporal_kernel // 2
            self.stem = nn.Sequential(
                nn.Conv1d(
                    market_feature_count,
                    temporal_channels,
                    kernel_size=temporal_kernel,
                    stride=temporal_stride,
                    padding=self.temporal_padding,
                    bias=False,
                ),
                nn.GroupNorm(normalization_groups, temporal_channels),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                [ResidualTemporalBlock(temporal_channels, dilation) for dilation in temporal_dilations]
            )
            self.output = nn.Sequential(
                nn.Linear(temporal_channels, stock_embedding_dim),
                nn.LayerNorm(stock_embedding_dim),
                nn.GELU(),
            )

        def forward(self, sequences, history_mask):
            # sequences [market_date, stock, history, raw_feature]
            market_dates, stock_count, history_bars, raw_features = sequences.shape
            if raw_features != market_feature_count - 1:
                raise ValueError(
                    "market raw feature count 不一致: "
                    f"expected={market_feature_count - 1}, actual={raw_features}"
                )
            if history_mask.shape != (market_dates, stock_count, history_bars):
                raise ValueError("market history mask shape 不一致")
            mask_float = history_mask.to(dtype=sequences.dtype).unsqueeze(-1)
            augmented = torch.cat((sequences * mask_float, mask_float), dim=-1)
            flattened = augmented.reshape(
                market_dates * stock_count,
                history_bars,
                market_feature_count,
            ).transpose(1, 2)
            flattened_mask = history_mask.reshape(
                market_dates * stock_count,
                1,
                history_bars,
            ).to(dtype=sequences.dtype)
            z = self.stem(flattened)
            pooled_mask = torch.nn.functional.max_pool1d(
                flattened_mask,
                kernel_size=self.temporal_kernel,
                stride=self.temporal_stride,
                padding=self.temporal_padding,
            )
            valid_bins = pooled_mask > 0.0
            z = z * valid_bins.to(dtype=z.dtype)
            for block in self.blocks:
                z = block(z)
                z = z * valid_bins.to(dtype=z.dtype)
            denominator = valid_bins.to(dtype=z.dtype).sum(dim=2).clamp(min=1.0)
            pooled = (z * valid_bins.to(dtype=z.dtype)).sum(dim=2) / denominator
            return self.output(pooled).reshape(market_dates, stock_count, stock_embedding_dim)

    class LearnedMarketSetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stock_encoder = SharedStockEncoder()
            self.queries = nn.Parameter(
                torch.randn(1, query_count, stock_embedding_dim) * 0.02
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=stock_embedding_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.output = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(query_count * stock_embedding_dim, market_embedding_dim),
                nn.LayerNorm(market_embedding_dim),
                nn.GELU(),
            )

        def forward(self, sequences, history_mask, valid_stock_mask):
            embeddings = self.stock_encoder(sequences, history_mask)
            if valid_stock_mask.shape != embeddings.shape[:2]:
                raise ValueError("market valid_stock_mask shape 不一致")
            if torch.any(valid_stock_mask.sum(dim=1) == 0):
                raise ValueError("market set 至少需要一檔有效股票")
            queries = self.queries.expand(embeddings.shape[0], -1, -1)
            tokens, _weights = self.attention(
                query=queries,
                key=embeddings,
                value=embeddings,
                key_padding_mask=~valid_stock_mask,
                need_weights=False,
            )
            return self.output(tokens)

    class InceptionTimeMarketSetClassifier(nn.Module):
        requires_market_set = True

        def __init__(self):
            super().__init__()
            self.candidate_encoder = build_inception_time(
                nn,
                torch,
                feature_count=feature_count,
                context_count=context_count,
                spec=spec,
            )
            # The shared encoder representation is used directly; its 9A classification head
            # is not part of this architecture.
            self.candidate_encoder.classifier = nn.Identity()
            candidate_embedding_dim = int(spec.inception_filters) * (
                len(tuple(spec.inception_kernel_sizes)) + 1
            )
            self.market_encoder = LearnedMarketSetEncoder()
            self.fusion = nn.Sequential(
                nn.Linear(candidate_embedding_dim + market_embedding_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, 2),
            )

        def encode_candidate(self, x):
            return self.candidate_encoder.encode(x)

        def encode_market(self, sequences, history_mask, valid_stock_mask):
            return self.market_encoder(sequences, history_mask, valid_stock_mask)

        def fuse_embeddings(self, candidate_embedding, market_embedding):
            if candidate_embedding.ndim != 2 or market_embedding.ndim != 2:
                raise ValueError("candidate／market embedding 必須是 2D")
            if candidate_embedding.shape[0] != market_embedding.shape[0]:
                raise ValueError("candidate／market embedding batch size 不一致")
            return self.fusion(torch.cat((candidate_embedding, market_embedding), dim=1))

        def forward(self, x, context, market_inputs=None):
            del context
            if market_inputs is None or len(market_inputs) != 4:
                raise ValueError(
                    "Market Set model 需要 (sequences, history_mask, valid_stock_mask, event_to_market)"
                )
            sequences, history_mask, valid_stock_mask, event_to_market = market_inputs
            candidate_embedding = self.encode_candidate(x)
            market_by_date = self.encode_market(
                sequences,
                history_mask,
                valid_stock_mask,
            )
            if event_to_market.ndim != 1 or event_to_market.shape[0] != x.shape[0]:
                raise ValueError("event_to_market shape 與 candidate batch 不一致")
            market_embedding = market_by_date[event_to_market]
            return self.fuse_embeddings(candidate_embedding, market_embedding)

    return InceptionTimeMarketSetClassifier()


__all__ = ["build_inception_time_market_set"]
