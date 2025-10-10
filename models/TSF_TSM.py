import torch
import torch.nn as nn
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from layers.TSF_TSM_backbone import *
from layers.TSF_TSM_layers import *
from layers.TSF_TSM_adaptive_blocks import AdaptiveNormalizationBlock
from layers.TSF_TSM_experts_blocks import ContextEncoder
from utils.metrics import quantile_loss, energy_score

torch.manual_seed(42)
np.random.seed(42)

class Model(nn.Module):
    """
    최종 제안 모델: 적응형 정규화, MoE 인코더, 그리고 
    결정론적 헤드 + NF 기반 확률론적 헤드를 결합한 아키텍처.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.is_training = configs.is_training
        
        self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        self.encoder = ContextEncoder(configs)
        self.mean_head = MeanPredictionHead(configs)
        self.residual_head = ProbabilisticResidualModel(configs) 
        self.context_fuse = nn.Linear(self.d_model * 2, self.d_model)

        # 손실 함수 및 잔차 정규화 통계치
        self.register_buffer('residual_mean', torch.zeros(1, 1, configs.enc_in))
        self.register_buffer('residual_std', torch.ones(1, 1, configs.enc_in))

        # 모멘텀 값 (0.9 ~ 0.999 사이의 값 사용)
        self.momentum = configs.momentum
        self.ln = nn.LayerNorm(configs.enc_in)  # 배치 정규화를 위한 레이어

    def forward(self, x_enc, batch_y, epoch):
        y_true = batch_y[:, -self.pred_len:, :]

        # 1. Adaptive normalization
        normalized_x, means, stdev, trend, norm_context = self.adaptive_norm_block.normalize(x_enc)

        y_true_detrended = y_true - trend[:, -y_true.size(1):, :]
        normalized_y_true = (y_true_detrended - means) / stdev

        # 2. Context encoder
        summary_context = self.context_fuse(torch.cat([self.encoder(normalized_x), norm_context], dim=-1))

        # 3. Deterministic mean prediction
        mean_pred_norm = self.mean_head(summary_context)

        # --- Stable residual learning ---
        residual_input = normalized_y_true - mean_pred_norm
        residuals_norm = self.ln(residual_input)

        # 잔차 통계 갱신
        if self.is_training:
            batch_mean = residuals_norm.mean().detach()   # scalar
            batch_std  = residuals_norm.std().detach()    # scalar
            self.residual_mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
            self.residual_std.mul_(self.momentum).add_(batch_std * (1 - self.momentum))

        scaled_residuals = (residuals_norm - self.residual_mean) / (self.residual_std + 1e-8)

        # 5. Losses
        huber = F.smooth_l1_loss(mean_pred_norm, normalized_y_true)
        q10 = quantile_loss(mean_pred_norm, normalized_y_true, 0.1)
        q90 = quantile_loss(mean_pred_norm, normalized_y_true, 0.9)
        mean_loss = huber + q10 + q90

        resi_loss = self.residual_head(summary_context, y=scaled_residuals)

        return mean_loss, resi_loss

    def sample(self, x_enc, num_samples=5):
        # --- 추론(Inference) 과정 ---
        self.eval()
        with torch.no_grad():
            # 1. 정규화 및 최종 컨텍스트 생성 (기존과 동일)
            normalized_x, means, stdev, trend, norm_context = self.adaptive_norm_block.normalize(x_enc)
            summary_context = self.context_fuse(torch.cat([self.encoder(normalized_x), norm_context], dim=-1))
            
            # 2. 평균 예측 (mean_head)
            mean_pred_norm = self.mean_head(summary_context)
            
            # 3. ✅ [핵심] 잔차(Residual) 샘플링 (residual_head)
            residual_samples_scaled = self.residual_head.sample(summary_context, num_samples=num_samples).mean().squeeze(0)
            residual_samples_norm = (residual_samples_scaled * self.residual_std) + self.residual_mean
            
            # 4. ✅ [핵심] 평균 예측과 잔차 샘플링 결합
            final_pred_norm = mean_pred_norm + residual_samples_norm
            
            # 5. 전체 스케일 복원 (De-normalization)
            trend_for_forecast = trend[:, -self.pred_len:, :]
            final_forecast = self.adaptive_norm_block.denormalize(final_pred_norm, means, stdev, trend_for_forecast)
            
            return final_forecast
