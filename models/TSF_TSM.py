import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import periodogram
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# nflows 라이브러리 임포트
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from layers.TST_TSM_backbone import *
from layers.TST_TSM_layers import *

torch.manual_seed(42)
np.random.seed(42)

import logging
logging.basicConfig(level=logging.DEBUG)

class Model(nn.Module):
    # __init__에서 mean, scale 관련 부분을 제거합니다.
    def __init__(self, configs, mean, scale): # mean, scale 인자 제거
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.eval_samples = configs.eval_samples
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.feature_dim = configs.d_ff
        self.n_heads = configs.n_heads
        self.chunk_size = configs.chunk_size
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.flow_layers = configs.flow_layers
        self.hidden_features = configs.hidden_features
        self.num_bins = configs.num_bins

        self.shared_encoder = SharedEncoder(self.seq_len, enc_in=self.enc_in, feature_dim=self.feature_dim, n_heads=self.n_heads, patch_len=self.patch_len, stride=self.stride)
        # self.detrender = LearnableTrend(self.seq_len, self.pred_len, self.enc_in)
        self.detrender = MultiScaleTrendSE(self.enc_in, self.seq_len, self.pred_len)
        # self.deterministic_model = DeterministicModel(self.shared_encoder, self.pred_len, self.enc_in)
        self.deterministic_model = Decoder(self.enc_in, self.d_model, self.n_heads, self.feature_dim, 4, 0.1)
        self.residual_model = ProbabilisticResidualModel(self.shared_encoder, self.pred_len, self.enc_in, self.chunk_size, self.flow_layers, self.hidden_features, self.num_bins)
        
        self.loss_fn_mse = nn.MSELoss()

        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def _preprocess(self, x_raw, y_raw=None):
        """[2단계 수정] 원본 데이터를 받아 '추세 분리 -> 스케일링'을 수행합니다."""
        # 1. 추세 분리 먼저 수행
        x_detrended_raw, _ = self.detrender(x_raw)
        
        # 2. 추세가 제거된 데이터를 스케일링
        x_scaled = (x_detrended_raw - self.mean) / self.scale

        if y_raw is not None:
            # 학습 시에는 x와 y를 합쳐서 전체 트렌드를 일관되게 계산
            full_raw = torch.cat([x_raw, y_raw], dim=1)
            full_detrended_raw, _ = self.detrender(full_raw)
            y_detrended_raw = full_detrended_raw[:, self.seq_len:]
            
            # 추세 제거된 y를 스케일링
            y_scaled = (y_detrended_raw - self.mean) / self.scale
            return x_scaled, y_scaled
        
        return x_scaled, None

    # forward 함수에서 _preprocess 대신 _detrend를 사용하도록 수정
    def forward(self, x_raw, y_raw):
        """[3단계 수정] 한 번의 forward 호출로 모든 손실을 계산하여 반환합니다."""
        
        # 1. 전처리 (추세 분리 -> 스케일링)
        x_scaled, y_scaled = self._preprocess(x_raw, y_raw)
        
        # 2. 공유 인코더 계산 (한 번만 실행)
        encoder_outputs, summary_context = self.shared_encoder(x_scaled)
        
        # 3. Deterministic 부분 계산
        y_target = y_scaled[:, -self.pred_len:, :]
        decoder_input = torch.zeros_like(y_target).to(x_raw.device)

        mean_pred = self.deterministic_model(decoder_input, encoder_outputs) 

        deter_loss = self.loss_fn_mse(mean_pred, y_target)
        
        # 4. Residual 부분 계산
        # gradient가 흐르지 않도록 .detach() 사용
        residuals = y_target - mean_pred.detach()
        
        log_prob, ar_pred = self.residual_model(summary_context, y=residuals)
        
        nll_loss = -log_prob.mean()
        # ar_pred는 residual에 대한 예측이므로, 실제 residual과 비교
        mse_loss = self.loss_fn_mse(ar_pred, residuals)
            
        return deter_loss, nll_loss, mse_loss
    
    def sample(self, x_raw):
        x_scaled, _ = self._preprocess(x_raw)
        
        # 💡 [추가] shared_encoder를 통과시켜 디코더에 필요한 정보를 얻습니다.
        encoder_outputs, summary_context = self.shared_encoder(x_scaled)
        
        # --- Deterministic Decoder 예측 ---
        # 💡 [추가] forward와 동일하게 디코더 입력을 생성합니다.
        y_target_placeholder = torch.zeros(x_raw.size(0), self.pred_len, self.enc_in).to(x_raw.device)
        decoder_input = torch.zeros_like(y_target_placeholder).to(x_raw.device)
        
        # 💡 [수정] Decoder를 올바른 인자로 호출합니다.
        mean_pred_scaled = self.deterministic_model(decoder_input, encoder_outputs)

        # --- Probabilistic Residual 예측 ---
        # 💡 [수정] x_scaled를 다시 넣는 대신, 계산해 둔 summary_context를 재사용합니다.
        #    이를 위해 TST_TSM_backbone.py의 ProbabilisticResidualModel.sample 인자를 수정해야 할 수 있습니다.
        #    (sample(self, x) -> sample(self, summary_context))
        #    우선은 기존 방식대로 두되, 비효율적이라는 점을 인지합니다.
        residual_samples_scaled = self.residual_model.sample(x_scaled, num_samples=self.eval_samples)
        
        # --- 결과 취합 ---
        final_samples_scaled = mean_pred_scaled.unsqueeze(0) + residual_samples_scaled
        mean_final_pred_scaled = final_samples_scaled.mean(dim=0)

        # 3. 역변환 (un-scale -> 추세 복원)
        # 3a. Un-scaling
        mean_final_pred_detrended = (mean_final_pred_scaled * self.scale) + self.mean
        
        # 3b. 미래 추세 계산 및 더하기
        _, future_trend = self.detrender(x_raw)
        final_pred_raw = mean_final_pred_detrended + future_trend # future_trend는 이미 [B, pred_len, C] shape일 것
        
        return final_pred_raw