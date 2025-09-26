import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 블록. 채널(피처)별 중요도를 학습합니다."""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Len]
        B, C, _ = x.shape
        y = self.squeeze(x).view(B, C) # Squeeze: [B, C]
        y = self.excitation(y).view(B, C, 1) # Excitation: [B, C, 1]
        return x * y.expand_as(x) # Scale (Re-calibrate)

class MultiScaleTrendSE(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, d_model, kernel_sizes=[3, 7, 15, 25], reduction_ratio=4):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_sizes = kernel_sizes
        self.trend_convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        num_combined_channels = self.enc_in * len(kernel_sizes)
        self.se_block = SEBlock(num_combined_channels, reduction_ratio=reduction_ratio)
        self.projection = nn.Linear(num_combined_channels, self.enc_in)

        self.context_projection = nn.Linear(num_combined_channels, d_model)
        self.context_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x_transposed = x.permute(0, 2, 1)
        trend_outputs = [conv(x_transposed) for conv in self.trend_convs]
        concatenated_trends = torch.cat(trend_outputs, dim=1)
        pooled_features = self.context_pool(concatenated_trends).squeeze(-1) # [B, C * num_kernels]
        norm_context = self.context_projection(pooled_features) # [B, d_model]
        recalibrated_trends = self.se_block(concatenated_trends)
        recalibrated_trends = recalibrated_trends.permute(0, 2, 1)
        final_trend = self.projection(recalibrated_trends)
        detrended_x = x - final_trend
        return detrended_x, final_trend, norm_context

class AdaptiveNormalizationBlock(nn.Module):
    """Detrender와 NormalizationHead를 결합한 적응형 정규화 블록."""
    def __init__(self, configs):
        super().__init__()
        self.detrender_context_generator = MultiScaleTrendSE(configs.enc_in, configs.seq_len, configs.pred_len, configs.d_model)

    def normalize(self, x):
        # x shape: [B, L, C]
        # context shape: [B, d_model]
        detrended_x, trend, _ = self.detrender_context_generator(x)
        means = detrended_x.mean(dim=[0, 1], keepdim=True).detach()
        stdev = torch.sqrt(torch.var(detrended_x, dim=[0, 1], keepdim=True, unbiased=False) + 1e-5)
        normalized_x = (detrended_x - means) / stdev

        # 역정규화를 위해 필요한 모든 값을 반환
        return normalized_x, means, stdev, trend

    def denormalize(self, y_norm, means, stdev, trend):
        # 1. Instance Normalization을 되돌림
        y_detrended = y_norm * stdev + means
        
        # 2. 제거했던 트렌드를 다시 더함
        final_y = y_detrended + trend
        
        return final_y