import math
import torch
import torch.nn as nn

# 使用绝对导入
from src.VisionFM.models.vision_transformer import vit_base

# 使用绝对导入
from src.utils import load_pretrained_weights

class VFMImageSurvivalModel(nn.Module):
    """
    Single-image survival model using VisionFM as the encoder.
    This is a direct replacement for the ImageSurvivalModel class but using VFM's ViT.
    """
    def __init__(self, args):
        super(VFMImageSurvivalModel, self).__init__()
        
        # Initialize VFM encoder with specified patch size and no classification head
        self.encoder = vit_base(patch_size=args.vfm_patch_size, num_classes=0)
        
        # Load pretrained weights
        load_pretrained_weights(
            model=self.encoder,
            pretrained_weights=args.vfm_checkpoint_path,
            checkpoint_key=args.vfm_checkpoint_key,
            model_name='vit_base',
            patch_size=args.vfm_patch_size
        )
        
        # Freeze encoder parameters if specified
        if getattr(args, 'freeze_encoder', True):
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Define classifier head for survival prediction
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(self.encoder.embed_dim, args.n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features using VFM's get_intermediate_layers method
        # This returns the output of the last Transformer block
        features = self.encoder.get_intermediate_layers(x, n=1)[0]
        
        # Use the CLS token for classification
        cls_token = features[:, 0]
        
        # Predict hazards
        hazards = self.classifier(cls_token)
        
        # Compute survival probabilities
        surv = torch.cumprod(1-hazards, dim=1)
        
        return hazards, surv

class VFM_LTSA(nn.Module):
    """
    Longitudinal Transformer for Survival Analysis using VisionFM as the image encoder.
    This is a modified version of the LTSA class that uses VFM's ViT as the image encoder.
    """
    def __init__(self, args):
        super(VFM_LTSA, self).__init__()
        self.args = args
        
        # Initialize VFM encoder with specified patch size and no classification head
        self.encoder = vit_base(patch_size=args.vfm_patch_size, num_classes=0)
        
        # Load pretrained weights
        load_pretrained_weights(
            model=self.encoder,
            pretrained_weights=args.vfm_checkpoint_path,
            checkpoint_key=args.vfm_checkpoint_key,
            model_name='vit_base',
            patch_size=args.vfm_patch_size
        )
        
        # Freeze encoder parameters if specified
        if getattr(args, 'freeze_encoder', True):
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # The feature dimension is the embedding dimension of the ViT
        feature_dim = self.encoder.embed_dim  # 768 for vit_base
        
        # Define transformer components for temporal modeling
        if args.attn_map:
            # Use custom TransformerEncoder + TransformerEncoderLayer to allow attention map extraction
            from my_transformers import TransformerEncoderLayer, TransformerEncoder

            transformer_encoder = TransformerEncoderLayer(d_model=feature_dim, nhead=args.n_heads, dim_feedforward=feature_dim, dropout=args.dropout, activation='relu', batch_first=True)
            self.transformer = TransformerEncoder(transformer_encoder, num_layers=args.n_layers)
        else:
            transformer_encoder = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=args.n_heads, dim_feedforward=feature_dim, dropout=args.dropout, activation='relu', batch_first=True)
            self.transformer = nn.TransformerEncoder(transformer_encoder, num_layers=args.n_layers)

        # Position encoding components
        if args.learned_pe:
            self.pos_encoder = LearnedPositionalEncoding(d_model=feature_dim)
        else:
            if args.tpe: 
                self.pos_encoder = PositionalEncoding(d_model=feature_dim, dropout=0, max_len=args.max_seq_len*12)  # 12 since measured in months
            else:
                self.pos_encoder = PositionalEncoding(d_model=feature_dim, dropout=0, max_len=args.max_seq_len)

        if args.amd_sev_enc:
            self.amd_sev_encoder = PositionalEncoding(d_model=feature_dim, dropout=0, max_len=args.max_seq_len)
        else:
            self.amd_sev_encoder = None

        # Classifier head for survival prediction
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(feature_dim, args.n_classes),
            nn.Sigmoid()
        )

        # Step-ahead prediction components
        if args.step_ahead:
            self.rel_pos_encoder = PositionalEncoding(d_model=feature_dim, dropout=0, max_len=args.max_seq_len*12)
            self.step_ahead_predictor = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(feature_dim, feature_dim)
            )

        # Causal mask for transformer
        self.causal_mask = torch.triu(torch.full((args.max_seq_len, args.max_seq_len), float('-inf'), device='cuda:0'), diagonal=1)

    def forward(self, x, seq_lengths, rel_times, prior_AMD_sevs):
        # x shape: batch_size*max_seq_len x 3 x H x W
        batch_size = len(seq_lengths)
        
        # Extract features using VFM's get_intermediate_layers method
        features = self.encoder.get_intermediate_layers(x, n=1)[0]
        
        # Use the CLS token for each image
        cls_tokens = features[:, 0]  # batch_size*max_seq_len x embed_dim
        
        # Reshape embeddings to batch_size x seq_length x embed_dim
        embeddings = cls_tokens.reshape(batch_size, self.args.max_seq_len, self.encoder.embed_dim)

        # Apply positional encoding
        if self.args.tpe:
            x = self.pos_encoder(embeddings, rel_times)
        else:
            x = self.pos_encoder(embeddings, None)

        # Apply AMD severity encoding if enabled
        if self.args.amd_sev_enc:
            x = self.amd_sev_encoder(x, prior_AMD_sevs)

        # Create mask to ignore padding tokens. For each sequence of visits, mask all tokens beyond last visit
        # Here, 1 = pad (ignore), 0 = valid (keep)
        src_key_padding_mask = torch.ones((x.shape[0], x.shape[1])).float().to('cuda:0')  # batch x seq_length
        for i, seq_length in enumerate(seq_lengths):
            src_key_padding_mask[i, :seq_length] = 0

        # Transformer modeling with "decoder-style" causal attention
        if self.args.attn_map:
            feats, attn_map = self.transformer(x, mask=self.causal_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True, need_weights=self.args.attn_map)
        else:
            feats = self.transformer(x, mask=self.causal_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True)
            
        # Re-pad sequences if needed
        if feats.shape[1] < self.args.max_seq_len:  # b x seq x feat
            feats = nn.functional.pad(feats, (0, 0, 0, self.args.max_seq_len - feats.shape[1], 0, 0), mode='constant')

        # Predict discrete-time hazard distribution
        hazards = self.classifier(feats)
        
        # Generate discrete-time survival probabilities
        surv = torch.cumprod(1-hazards.view(-1, hazards.shape[-1]), dim=1).view(hazards.shape[0], hazards.shape[1], hazards.shape[2])

        # Padding mask used to compute loss later
        padding_mask = torch.bitwise_not(src_key_padding_mask.bool()).unsqueeze(-1)

        # Step-ahead prediction if enabled
        if self.args.step_ahead:
            # Get time elapsed (delta) between consecutive visits
            delta_times = torch.diff(rel_times)  # batch x max_seq_len-1
            delta_times[delta_times < 0] = 0
            delta_times = nn.functional.pad(delta_times, (0, 1), 'constant', 0)  # batch x max_seq_len

            # Use relative temporal timestep encoding
            delta_encoded_feats = self.rel_pos_encoder(feats, delta_times) 

            # Predict imaging features of *next* visit for each subsequence
            feat_preds = self.step_ahead_predictor(delta_encoded_feats)

            # Get actual imaging features of next visit
            feat_targets = nn.functional.pad(feats[:, 1:, :], (0, 0, 0, 1), 'constant', 0)

            if self.args.attn_map:
                return hazards, surv, feat_preds, feat_targets, padding_mask, attn_map
            else:
                return hazards, surv, feat_preds, feat_targets, padding_mask

        if self.args.attn_map:
            return hazards, surv, padding_mask, attn_map
        else:
            return hazards, surv, padding_mask

# Copy of the PositionalEncoding class from models_long.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Switch dimensions to BATCH FIRST (seq x batch x feat -> batch x seq x feat)
        self.pe = torch.permute(self.pe, (1, 0, 2))

    def forward(self, x, rel_times):
        if rel_times is None:
            x = x + self.pe[0, :x.size(1), :]
        else:
            for i, t in enumerate(rel_times):
                x[i, :, :] += self.pe[0, t, :]

        return self.dropout(x)

# Copy of the LearnedPositionalEncoding class from models_long.py
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Sequential(
            nn.Linear(1, int(math.sqrt(d_model))),
            nn.Tanh(),
            nn.Linear(int(math.sqrt(d_model)), d_model)
        )

    def forward(self, x, times):
        # x: batch x max_seq_len x 512
        # times: batch x max_seq_len
        times = times.unsqueeze(-1).float().to(x.device)

        # time_embeddings: batch x max_seq_len x 512
        time_embeddings = self.pe(times)

        return x + time_embeddings 