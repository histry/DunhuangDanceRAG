import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalMMR(nn.Module):
    def __init__(self, motion_dim=151, audio_dim=803, latent_dim=256, num_bins=4):
        super().__init__()
        self.num_bins = num_bins
       # 【修复4】：将输入维度扩展为特征维度乘以时间块数 (e.g., 803 * 4)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim * num_bins, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )
        
        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_dim * num_bins, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

    def encode_audio(self, audio_feat):
        """
        audio_feat: [batch_size, seq_len, audio_dim]
        """
        # 转置为 [batch_size, audio_dim, seq_len] 以适配 1D 池化
        feat_t = audio_feat.transpose(1, 2)
        # 自适应池化为 num_bins 个时间块
        pooled = F.adaptive_avg_pool1d(feat_t, self.num_bins)
        # 展平特征维和时间块维: [batch_size, audio_dim * num_bins]
        flattened = pooled.flatten(start_dim=1)
        return self.audio_encoder(flattened)

    def encode_motion(self, motion_feat):
        """
        motion_feat: [batch_size, seq_len, motion_dim]
        """
        feat_t = motion_feat.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(feat_t, self.num_bins)
        flattened = pooled.flatten(start_dim=1)
        return self.motion_encoder(flattened)