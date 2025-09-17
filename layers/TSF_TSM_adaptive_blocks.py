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
            nn.ReLU(inplace=True),
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
        return detrended_x, norm_context

class NormalizationHead(nn.Module):
    """컨텍스트를 기반으로 정규화 파라미터(scale, shift)를 예측하는 헤드."""
    def __init__(self, d_model, c_in):
        super().__init__()
        self.scale_head = nn.Linear(d_model, c_in)
        self.shift_head = nn.Linear(d_model, c_in)

    def forward(self, context):
        # context shape: [B, d_model]
        # Softplus를 통해 scale이 항상 양수가 되도록 보장
        scale = F.softplus(self.scale_head(context)) + 1e-6 # [B, c_in]
        shift = self.shift_head(context) # [B, c_in]
        return scale, shift

class AdaptiveNormalizationBlock(nn.Module):
    """Detrender와 NormalizationHead를 결합한 적응형 정규화 블록."""
    def __init__(self, configs):
        super().__init__()
        self.detrender_context_generator = MultiScaleTrendSE(configs.enc_in, configs.seq_len, configs.pred_len, configs.d_model)
        self.normalization_head = NormalizationHead(configs.d_model, configs.enc_in)

    def normalize(self, x):
        # x shape: [B, L, C]
        # context shape: [B, d_model]
        detrended_x, norm_context = self.detrender_context_generator(x)
        scale, shift = self.normalization_head(norm_context)

        # 정규화를 위해 scale/shift를 [B, 1, C]로 차원 확장
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        normalized_x = (detrended_x - shift) / scale
        return normalized_x, scale, shift

    def denormalize(self, y, scale, shift):
        # y shape: [B, L_pred, C]
        # scale/shift shape: [B, 1, C]
        denormalized_y = y * scale + shift
        return denormalized_y