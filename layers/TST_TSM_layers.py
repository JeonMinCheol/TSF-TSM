import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging

# nflows 라이브러리 임포트
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet

def create_conditional_nsf_flow(feature_dim, context_dim, num_layers=5, hidden_features=128, num_bins=8):
    transforms = []
    mask = torch.zeros(feature_dim)
    mask[:feature_dim // 2] = 1

    for i in range(num_layers):
        transforms.append(RandomPermutation(features=feature_dim))
        current_mask = mask if i % 2 == 0 else 1 - mask
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=current_mask,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_blocks=2,
                    activation=F.relu,
                ),
                num_bins=num_bins, tails="linear", tail_bound=5.0
            )
        )
        transforms.append(BatchNorm(features=feature_dim))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([feature_dim])
    )


class PatchingEmbedding(nn.Module):
    def __init__(self, patch_len, stride, c_in, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # 각 패치를 d_model 차원의 벡터로 변환할 Linear 레이어
        # 패치는 (patch_len * c_in) 크기의 1D 벡터로 펼쳐집니다.
        self.projection = nn.Linear(patch_len * c_in, d_model)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        # 예: [512, 336, 7]
        
        # 1. 시퀀스를 패치 단위로 자릅니다. (stride 만큼 겹치면서)
        # unfold: 텐서를 슬라이딩 윈도우 방식으로 잘라주는 매우 효율적인 함수
        # (B, L, C) -> (B, C, L) -> unfold -> (B, C, Num_Patches, Patch_Len)
        x_unfolded = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 2. 패치를 펼치기(flatten) 위해 차원을 재정렬하고 합칩니다.
        # (B, C, Num_Patches, Patch_Len) -> (B, Num_Patches, C, Patch_Len)
        x_patched = x_unfolded.permute(0, 2, 1, 3)
        
        # (B, Num_Patches, C, Patch_Len) -> (B, Num_Patches, C * Patch_Len)
        B, n_patches, C, P = x_patched.shape
        x_flattened = x_patched.reshape(B, n_patches, -1)
        
        # 3. Linear 레이어를 통과시켜 각 패치를 d_model 차원의 벡터로 임베딩
        # (B, Num_Patches, C * Patch_Len) -> (B, Num_Patches, d_model)
        out = self.projection(x_flattened)
        
        return out
        
"""
Summary context -> predicted correction sequence for residuals.
We subtract this from residuals before feeding flow (so flow models the 'whitened' residual).
During sampling we add this AR prediction back to the sampled residuals.
"""
class ResidualAR(nn.Module):
    def __init__(self, context_dim, pred_len, enc_in, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, pred_len * enc_in)
        )
    def forward(self, context):
        # context: (B, context_dim) -> output (B, pred_len)
        return self.net(context)


"""
MLP를 사용하여 데이터에서 비선형 추세를 학습하고 분리하는 모듈
"""
class LearnableTrend(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 시간 인덱스를 입력받아 추세 값을 출력하는 간단한 MLP
        self.trend_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, enc_in)
        )

    def forward(self, x_batch):
        # x_batch shape: (B, current_seq_len, enc_in)
        
        # 💡 [수정] 배치 크기(B)를 입력 텐서로부터 가져옵니다.
        B, current_seq_len, _ = x_batch.shape
        
        # 2. 현재 길이에 맞는 시간 인덱스 벡터를 생성합니다.
        past_time_idx = torch.arange(current_seq_len, device=x_batch.device, dtype=torch.float32)
        past_time_idx_normalized = (past_time_idx / (self.seq_len - 1)).unsqueeze(-1)

        # 3. MLP를 통과시켜 과거 추세를 계산합니다.
        #    - unsqueeze(0)를 통해 브로드캐스팅이 가능하도록 배치 차원을 추가합니다.
        past_trend = self.trend_mlp(past_time_idx_normalized).unsqueeze(0)
        
        # 4. 원본 데이터에서 추세를 분리합니다.
        detrended_x = x_batch - past_trend
        
        # 5. 예측할 미래 시점에 대한 추세를 계산합니다.
        #    - `past_time_idx` 대신 `current_seq_len`을 사용하면 더 명확합니다.
        future_time_idx = torch.arange(self.seq_len, self.seq_len + self.pred_len, device=x_batch.device, dtype=torch.float32)
        future_time_idx_normalized = (future_time_idx / (self.seq_len - 1)).unsqueeze(-1)
        
        future_trend = self.trend_mlp(future_time_idx_normalized)

        # 💡 [수정] future_trend를 배치 크기에 맞게 expand 해줍니다.
        #    - (pred_len, enc_in) -> (1, pred_len, enc_in) -> (B, pred_len, enc_in)
        future_trend = future_trend.unsqueeze(0).expand(B, -1, -1)
        
        return detrended_x, future_trend

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # 1. Masked Self-Attention (디코더가 미래를 보지 못하게 함)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        # 2. Cross-Attention (디코더가 인코더의 출력을 참고함)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        # 3. Feed-Forward Network
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None):
        # x: 디코더 입력 [B, T_tgt, C]
        # memory: 인코더 출력 [B, T_src, C]
        
        # 1. Masked Self-Attention
        x_attn, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(x_attn))
        
        # 2. Cross-Attention
        x_cross_attn, _ = self.cross_attention(query=x, key=memory, value=memory)
        x = self.norm2(x + self.dropout(x_cross_attn))
        
        # 3. Feed-Forward
        x_ffn = self.ffn(x)
        x = self.norm3(x + self.dropout(x_ffn))
        
        return x

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
    """멀티스케일 Conv1d와 SE 블록을 결합한 추세 모델."""
    def __init__(self, enc_in, seq_len, pred_len, kernel_sizes=[3, 7, 15, 25], reduction_ratio=4):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_sizes = kernel_sizes
        
        # 1. 여러 개의 Conv1d 레이어를 담을 ModuleList
        self.trend_convs = nn.ModuleList()
        for k in self.kernel_sizes:
            self.trend_convs.append(
                nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=k, padding='same')
            )
            
        # 2. 합쳐진 추세들의 중요도를 학습할 SE 블록
        num_combined_channels = self.enc_in * len(self.kernel_sizes)
        self.se_block = SEBlock(num_combined_channels, reduction_ratio=reduction_ratio)
            
        # 3. SE 블록을 거친 추세들을 원래 차원으로 되돌릴 Linear 레이어
        self.projection = nn.Linear(num_combined_channels, self.enc_in)

    def forward(self, x):
        # x shape: [B, L, C] (Batch, Seq_Len, Channels/Features)
        B, L, C = x.shape
        
        # Conv1d에 맞게 shape 변경: [B, C, L]
        x_transposed = x.permute(0, 2, 1)
        
        # 각 Conv 레이어를 통과시켜 다른 스케일의 추세 추출
        trend_outputs = []
        for conv in self.trend_convs:
            trend_outputs.append(conv(x_transposed))
            
        # 모든 추세 결과를 채널(C) 기준으로 합치기
        concatenated_trends = torch.cat(trend_outputs, dim=1) # shape: [B, C * num_kernels, L]
        
        # SE 블록을 통해 채널별(스케일별) 중요도 적용
        recalibrated_trends = self.se_block(concatenated_trends)
        
        # Linear 레이어에 맞게 shape 복원: [B, L, C * num_kernels]
        recalibrated_trends = recalibrated_trends.permute(0, 2, 1)
        
        # 최종 추세 계산
        final_trend = self.projection(recalibrated_trends) # shape: [B, L, C]
        
        # 원본 데이터에서 추세 분리
        detrended_x = x - final_trend
        
        # 미래 추세 예측 (간단한 버전: 마지막 추세 값을 반복)
        # 더 정교하게 만들려면 이 부분에 MLP 등을 추가할 수 있습니다.
        future_trend = final_trend[:, -1:, :].repeat(1, self.pred_len, 1)
        
        return detrended_x, future_trend