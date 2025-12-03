import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, "/home/lin01231/zhan9191/AI4M/SongProj/deformable_detr")
from transformers import TimesformerModel, TimesformerConfig
from models.ops.modules.ms_deform_attn import MSDeformAttn

class SimpleConvEmbed(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        self.n_features = d_model

    def forward(self, x, seq_lengths=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)         # → [B*T, C, H, W]
        feat = self.conv(x)               # → [B*T, D, H', W']
        feat = feat.flatten(2).mean(-1)   # → [B*T, D]
        return feat.view(B, T, -1)        # → [B, T, D]
class TransformerImageEncoder(nn.Module):
    def __init__(self, conv_embed, transformer):
        super().__init__()
        self.conv_embed = conv_embed
        self.transformer = transformer
        self.n_features = conv_embed.n_features

    def forward(self, x, seq_lengths=None):
        x = self.conv_embed(x)
        return self.transformer(x)
        
def create_model(args):
    if args.use_deformable_spatial and args.use_deformable_temporal:
        encoder = DeformableSpatialEncoder(d_model=768, n_heads=4, n_points=4)
        temporal_encoder = DeformableTemporalEncoder(d_model=768, n_heads=4, max_len=args.max_seq_len)

    elif args.use_deformable_spatial and not args.use_deformable_temporal:
        encoder = DeformableSpatialEncoder(d_model=768, n_heads=4, n_points=4)
        temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True),
            num_layers=1
        )

    elif not args.use_deformable_spatial and args.use_deformable_temporal:
        conv_embed = SimpleConvEmbed(d_model=768)
        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True),
            num_layers=1
        )
        encoder = TransformerImageEncoder(conv_embed, transformer)
        encoder.n_features = 768
        temporal_encoder = DeformableTemporalEncoder(d_model=768, n_heads=4, max_len=args.max_seq_len)

    else:
        encoder = TimeSformerEncoder(pretrained=True, use_cls=not args.mean_pool)
        temporal_encoder = None  # TimeSformer already models both

    model = SF(
        encoder=encoder,
        temporal_encoder=temporal_encoder,
        args=args
    )
    return model

class DeformableSpatialEncoder(nn.Module):
    def __init__(self, d_model=768, n_heads=4, n_points=4, H=224, W=224):
        super().__init__()
        self.embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        self.attn = MSDeformAttn(d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
        self.attn.im2col_step = 1 
        self.proj = nn.Linear(d_model, d_model)

        self.H_feat = H // 16
        self.W_feat = W // 16
        self.spatial_shapes = torch.tensor([[self.H_feat, self.W_feat]], dtype=torch.long)
        self.level_start_index = torch.tensor([0], dtype=torch.long)
        self.n_features = d_model

    def forward(self, x, seq_lengths=None):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.embed(x)
        feat = feat.flatten(2).transpose(1, 2)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, self.H_feat, device=x.device),
            torch.linspace(0, 1, self.W_feat, device=x.device),
            indexing='ij'
        )
        ref_points = torch.stack((grid_x, grid_y), -1).view(-1, 2)
        reference_points = ref_points[None, :, None, :].repeat(B * T, 1, 1, 1)

        out = self.attn(
            query=feat,
            reference_points=reference_points,
            input_flatten=feat,
            input_spatial_shapes=self.spatial_shapes.to(x.device),
            input_level_start_index=self.level_start_index.to(x.device)
        )
        out = self.proj(out.mean(dim=1))
        return out.view(B, T, -1)

class DeformableTemporalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_points=4, num_layers=1, max_len=14):
        super().__init__()
        self.layers = nn.ModuleList([
            MSDeformAttn(d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
            for _ in range(num_layers)
        ])
        for layer in self.layers:
            layer.im2col_step = 1

        self.proj = nn.Linear(d_model, d_model)
        self.level_start_index = torch.tensor([0], dtype=torch.long)
        self.spatial_shapes = torch.tensor([[max_len, 1]], dtype=torch.long)

    def forward(self, x):
        B, T, D = x.shape
        query = x
        value = x

        ref = torch.linspace(0, 1, T, device=x.device)
        ref_points = torch.stack([ref, torch.zeros_like(ref)], dim=-1).unsqueeze(0).repeat(B, 1, 1)
        reference_points = ref_points.unsqueeze(2)

        for attn in self.layers:
            x = attn(
                query=query,
                reference_points=reference_points,
                input_flatten=value,
                input_spatial_shapes=self.spatial_shapes.to(x.device),
                input_level_start_index=self.level_start_index.to(x.device)
            )

        return self.proj(x)

class TimeSformerEncoder(nn.Module):
    def __init__(self, pretrained=True, use_cls=True):
        super().__init__()
        config = TimesformerConfig(num_frames=14)
        self.model = TimesformerModel(config)
        self.n_features = config.hidden_size
        self.use_cls = use_cls

    def forward(self, x, seq_lengths):
        B, T, C, H, W = x.shape
        outputs = self.model(pixel_values=x)

        if self.use_cls:
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings.unsqueeze(1).repeat(1, T, 1)
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1, keepdim=True).repeat(1, T, 1)

        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, learnable=False):
        super().__init__()
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)
        if learnable:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe_static', pe.permute(1, 0, 2))

    def forward(self, x, rel_times):
        B, T, D = x.shape

        if self.learnable:
            pe = self.pe(rel_times.to(x.device))
        else:
            if self.pe_static.size(1) < T:
                raise ValueError(f"Static positional encoding too short: pe has {self.pe_static.size(1)} steps, but x has {T}")
            pe = self.pe_static[0, :T, :]
            pe = pe.unsqueeze(0).expand(B, -1, -1)

        assert x.shape == pe.shape, f"Mismatch: x={x.shape}, pe={pe.shape}"
        return self.dropout(x + pe)

class SF(nn.Module):
    def __init__(self, encoder, temporal_encoder, args):
        super().__init__()
        self.encoder = encoder
        self.temporal_encoder = temporal_encoder
        self.args = args

        self.pos_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len, learnable=args.learned_pe)
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(self.encoder.n_features, args.n_classes),
            nn.Sigmoid()
        )

        if args.step_ahead:
            self.rel_pos_encoder = PositionalEncoding(d_model=self.encoder.n_features, dropout=0, max_len=args.max_seq_len*12, learnable=args.learned_pe)
            self.step_ahead_predictor = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(self.encoder.n_features, self.encoder.n_features)
            )
    def _generate_padding_mask(self, seq_lengths, max_len):
        B = len(seq_lengths)
        mask = torch.ones(B, max_len, dtype=torch.bool)
        for i, l in enumerate(seq_lengths):
            mask[i, :l] = False
        return mask
    def forward(self, x, seq_lengths, rel_times, prior_AMD_sevs):
        B, T, C, H, W = x.shape

        if isinstance(self.encoder, nn.TransformerEncoder):
            mask = self._generate_padding_mask(seq_lengths, x.shape[1]).to(x.device)
            embeddings = self.encoder(x, src_key_padding_mask=mask)
        else:
            embeddings = self.encoder(x, seq_lengths)
        x = self.pos_encoder(embeddings, rel_times)

        if self.temporal_encoder:
            x = self.temporal_encoder(x)

        hazards = self.classifier(x)
        surv = torch.cumprod(1 - hazards.view(-1, hazards.shape[-1]), dim=1).view(hazards.shape[0], hazards.shape[1], hazards.shape[2])

        padding_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        for i, l in enumerate(seq_lengths):
            padding_mask[i, :l] = 1
        padding_mask = padding_mask.unsqueeze(-1)

        if self.args.step_ahead:
            delta_times = torch.diff(rel_times, dim=1)
            delta_times = F.pad(delta_times, (0, 1), 'constant', 0)
            delta_encoded_feats = self.rel_pos_encoder(x, delta_times)
            feat_preds = self.step_ahead_predictor(delta_encoded_feats)
            feat_targets = F.pad(x[:, 1:, :], (0, 0, 0, 1), 'constant', 0)
            return hazards, surv, feat_preds, feat_targets, padding_mask

        return hazards, surv, padding_mask
