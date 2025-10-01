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

torch.manual_seed(42)
np.random.seed(42)

import logging

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

        self.adaptive_norm_block = AdaptiveNormalizationBlock(configs)
        self.encoder = ContextEncoder(configs)
        self.mean_head = MeanPredictionHead(configs)
        self.residual_head = ProbabilisticResidualModel(configs) 

        # 손실 함수 및 잔차 정규화 통계치
        self.loss_fn_mse = nn.MSELoss()

        self.register_buffer('residual_mean', torch.zeros(1))
        self.register_buffer('residual_std', torch.ones(1))

        # 모멘텀 값 (0.9 ~ 0.999 사이의 값 사용)
        self.momentum = configs.momentum
        self.ln = nn.LayerNorm(configs.enc_in)  # 배치 정규화를 위한 레이어

    def forward(self, x_enc, batch_y):
        y_true = batch_y[:, -self.pred_len:, :]

        # 1. 적응형 정규화
        normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
        y_true_detrended = y_true - trend
        normalized_y_true = (y_true_detrended - means) / stdev
        
        # 2. MoE 인코더를 통한 최종 특징 추출
        summary_context = self.encoder(normalized_x)

        # 3. 결정론적 헤드를 통한 평균 예측
        mean_pred_norm = self.mean_head(summary_context)
        
        # --- 안정화된 잔차 학습 (Stabilized Residual Learning) ---
        # A. 정규화된 공간에서의 실제 잔차 계산
        residuals_norm = self.ln(normalized_y_true - mean_pred_norm.detach())

        if self.training:
            # 현재 배치의 잔차 통계치 계산
            batch_mean = residuals_norm.mean().detach()
            batch_std = residuals_norm.std().detach()
            
            self.residual_mean = self.momentum * self.residual_mean + (1 - self.momentum) * batch_mean
            self.residual_std = self.momentum * self.residual_std + (1 - self.momentum) * batch_std

        scaled_residuals = (residuals_norm - self.residual_mean) / (self.residual_std + 1e-8)
        clipped_residuals = torch.clamp(scaled_residuals, min=-4.0, max=4.0)
        
        # D. 정규화된 잔차를 Normalizing Flow로 학습
        log_prob = self.residual_head(summary_context, y=clipped_residuals)
        deter_loss = self.loss_fn_mse(mean_pred_norm, normalized_y_true)
        nll_loss = -log_prob.mean()

        # 학습에 필요한 값들을 반환
        return deter_loss, nll_loss

    def sample(self, x_enc):
        # --- 추론(Inference) 과정 ---
        
        # 1. 정규화 및 최종 컨텍스트 생성
        normalized_x, means, stdev, trend = self.adaptive_norm_block.normalize(x_enc)
        summary_context = self.encoder(normalized_x)
        
        # 2. 평균 예측
        mean_pred_norm = self.mean_head(summary_context)
        
        # 3. NF 모델로 잔차 샘플링
        residual_samples_scaled = self.residual_head.sample(summary_context, num_samples=1).squeeze(0)
        
        # 4. 샘플링된 잔차를 원래 잔차 스케일로 복원
        residual_samples_norm = (residual_samples_scaled * self.residual_std) + self.residual_mean
        
        # 5. 최종 예측값 계산 (정규화된 공간에서)
        final_pred_norm = mean_pred_norm + residual_samples_norm
        
        # 6. 전체 스케일 복원 (De-normalization)
        final_forecast = self.adaptive_norm_block.denormalize(final_pred_norm, means, stdev, trend)
        
        return final_forecast