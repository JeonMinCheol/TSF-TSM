import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import periodogram
from .SelfAttention_Family import SparseVariationalAttention, AttentionPool
from .TST_TSM_layers import *

# nflows 라이브러리 임포트
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet

# --- 모델 구성 요소 ---

class ChunkedAutoregressiveFlowHead(nn.Module):
    def __init__(self, step_dim, context_dim, pred_len, chunk_size,
                 flow_layers=5, hidden_features=256, num_bins=16):
        super().__init__()
        self.step_dim = step_dim
        self.pred_len = pred_len
        self.chunk_size = chunk_size
        self.num_chunks = (pred_len + chunk_size - 1) // chunk_size

        self.flows = nn.ModuleList([
            create_conditional_nsf_flow(
                feature_dim=step_dim * chunk_size,
                context_dim=context_dim + step_dim * chunk_size * i,
                num_layers=flow_layers, hidden_features=hidden_features, num_bins=num_bins
            ) for i in range(self.num_chunks)
        ])

    def log_prob(self, inputs, context):
        # inputs의 기대 shape: (B, pred_len, step_dim)
        # 제공된 디버그 로그에서 shape가 (32, 96, 7)로 이미 올바르므로,
        # 추가적인 view 변환은 필요 없습니다.
        B = inputs.size(0)
        
        log_probs, prev_chunks_list = [], []
        for i, flow in enumerate(self.flows):
            flat_prev_chunks = torch.cat(prev_chunks_list, dim=-1) if prev_chunks_list else context.new_empty(B, 0)
            ar_context = torch.cat([context, flat_prev_chunks], dim=-1)
            
            start, end = i * self.chunk_size, min((i + 1) * self.chunk_size, self.pred_len)
            
            # ✅ 슬라이싱 후 view/reshape 전에 .contiguous()를 추가하여 메모리 관련 에러를 방지합니다.
            chunk_target = inputs[:, start:end, :].contiguous().view(B, -1)
            
            # 마지막 청크가 작을 경우, 크기를 맞춰주기 위한 패딩 처리
            if chunk_target.shape[1] < self.chunk_size * self.step_dim:
                padding = torch.zeros(B, self.chunk_size * self.step_dim - chunk_target.shape[1], device=inputs.device)
                chunk_target = torch.cat([chunk_target, padding], dim=1)

            log_probs.append(flow.log_prob(chunk_target, context=ar_context))
            
            # prev_chunks_list에 추가할 때도 동일하게 처리하여 일관성을 유지합니다.
            prev_chunks_list.append(chunk_target) # 이미 reshape 및 padding된 chunk_target을 사용

        return torch.stack(log_probs, dim=1).sum(dim=1)

    def sample(self, num_samples, context):
        B = context.size(0)
        all_samples, prev_chunks_list = [], []
        expanded_context = context.unsqueeze(0).expand(num_samples, B, -1).reshape(num_samples * B, -1)

        for i, flow in enumerate(self.flows):
            flat_prev_chunks = torch.cat(prev_chunks_list, dim=-1) if prev_chunks_list else expanded_context.new_empty(num_samples*B, 0)
            ar_context = torch.cat([expanded_context, flat_prev_chunks], dim=-1)
            
            # nflows의 sample 메서드는 contiguous 관련 문제가 거의 없지만, 일관성을 위해 그대로 둡니다.
            chunk_sample_flat = flow.sample(1, context=ar_context).squeeze(1)
            all_samples.append(chunk_sample_flat)
            prev_chunks_list.append(chunk_sample_flat)

        samples_padded = torch.cat(all_samples, dim=-1)
        # 패딩된 부분을 제거하고 원래 길이에 맞게 자릅니다.
        samples = samples_padded[:, :self.pred_len * self.step_dim]
        # 최종적으로 (num_samples, B, pred_len, step_dim) 형태로 복원합니다.
        return samples.view(num_samples, B, self.pred_len, self.step_dim)

class LearnableDecomposition(nn.Module):
    def __init__(self, sequence_length, period):
        super().__init__()
        self.trend_mlp = nn.Sequential(nn.Linear(sequence_length, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, sequence_length))
        self.seasonal_mlp = nn.Sequential(nn.Linear(sequence_length, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, sequence_length))
        self.period = period
        
    def forward(self, x):
        trend, seasonal = self.trend_mlp(x), self.seasonal_mlp(x)
        residual = x - trend - seasonal
        trend_loss = torch.mean((trend[:, 2:] - 2 * trend[:, 1:-1] + trend[:, :-2])**2)
        seasonal_loss = torch.mean((seasonal[:, self.period:] - seasonal[:, :-self.period])**2) if x.size(1) > self.period else torch.tensor(0.0, device=x.device)
        residual_loss = torch.mean((residual[:, 1:] * residual[:, :-1])**2)
        return trend, seasonal, residual, trend_loss, seasonal_loss, residual_loss


class ConditionalNFEncoder(nn.Module):
    def __init__(self, feature_dim, context_dim=1, num_flow_steps=3):
        super().__init__()
        self.mlp_trend = nn.Linear(1, feature_dim)
        self.mlp_seasonal = nn.Linear(1, feature_dim)
        transforms = []
        for _ in range(num_flow_steps):
            transforms.append(MaskedAffineAutoregressiveTransform(features=1, hidden_features=64, context_features=context_dim))
        self.residual_flow = Flow(CompositeTransform(transforms), StandardNormal(shape=[1]))
        
    def forward(self, trend, seasonal, residual):
        t, s, r = trend.unsqueeze(-1), seasonal.unsqueeze(-1), residual.unsqueeze(-1)
        feat_trend, feat_seasonal = self.mlp_trend(t), self.mlp_seasonal(s)

        B, T, D = r.shape
        flat_r = r.view(-1, D)
        context_flatten = torch.cat([torch.zeros_like(r[:, :1]), r[:, :-1]], dim=1).view(-1, D)

        with torch.no_grad():
            log_prob_residual = self.residual_flow.log_prob(flat_r, context=context_flatten).view(B, T, 1)
        return torch.cat([feat_trend, feat_seasonal, log_prob_residual], dim=-1)

import logging
# --- 2단계 아키텍처 ---
class SharedEncoder(nn.Module):
    def __init__(self, seq_len, enc_in, feature_dim=128, n_heads=4, n_pos_feats=6, patch_len=16, stride=8):
        super().__init__()
        self.final_feature_dim = feature_dim
        self.patching_embedding = PatchingEmbedding(patch_len, stride, enc_in, feature_dim)
        self.pos_proj = nn.Linear(n_pos_feats, self.final_feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.fusion = SparseVariationalAttention(feature_dim=self.final_feature_dim, n_heads=n_heads)
        self.attention_pool = AttentionPool(d_model=feature_dim, n_heads=4)

    def forward(self, x):
        features = self.patching_embedding(x)

        # [B, Num_Patches, d_model]
        B, T, C = features.shape 

        freqs = torch.tensor([1.0, 2.0, 4.0], device=x.device)  # choose small set of freqs
        pos = torch.arange(T, device=x.device).float().unsqueeze(0)  # (1,T)
        pos_feats = []

        for f in freqs:
            pos_feats.append(torch.sin(2 * np.pi * f * pos / T))
            pos_feats.append(torch.cos(2 * np.pi * f * pos / T))
        pos_feats = torch.stack(pos_feats, dim=-1).squeeze(0)  # (T, n_pos_feats)
        pos_feats = pos_feats.unsqueeze(0).expand(B, -1, -1)  # (B,T,n_pos_feats)
        pos_emb = self.pos_proj(pos_feats)  # (B,T,C)
        features = features + pos_emb
        
        context, _ = self.fusion(features)
        summary_context = context[:, -1, :]
        summary_context = self.attention_pool(context)
        
        return context, summary_context

class Decoder(nn.Module):
    def __init__(self, c_in, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        # 디코더 입력(이전 타임스텝의 출력)을 d_model 차원으로 임베딩
        self.embedding = nn.Linear(c_in, d_model)
        # 디코더 레이어를 여러 겹 쌓음
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        # 최종 출력을 위한 Linear 레이어
        self.projection = nn.Linear(d_model, c_in)

    def forward(self, x, memory):
        # x: 디코더의 입력 (예측 시작 토큰 등) [B, T_tgt, C_in]
        # memory: 인코더의 전체 시퀀스 출력 [B, T_src, d_model]
        
        x = self.embedding(x)
        
        # 디코더의 각 타임스텝이 미래를 보지 못하도록 마스크 생성
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
            
        return self.projection(x)
        
"""1단계: 평균 예측 모델"""
class DeterministicModel(nn.Module):
    def __init__(self, shared_encoder, pred_len, enc_in):
        super().__init__()
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.shared_encoder = shared_encoder
        self.head = nn.Linear(shared_encoder.final_feature_dim, self.pred_len * self.enc_in)

    def forward(self, x):
        summary_context = self.shared_encoder(x)
        mean_prediction_flat = self.head(summary_context) # shape: (B, pred_len * n_features)
        mean_prediction = mean_prediction_flat.view(-1, self.pred_len, self.enc_in)
        # logging.debug(f"5. mean_prediction: {mean_prediction.shape}")
        return mean_prediction


"""2단계: 잔차 분포 예측 모델"""
class ProbabilisticResidualModel(nn.Module):
    def __init__(self, shared_encoder, pred_len, step_dim, chunk_size, flow_layers, hidden_features, num_bins, ar_hidden=512):
        super().__init__()
        self.step_dim = step_dim
        self.shared_encoder = shared_encoder
        self.decoder = ChunkedAutoregressiveFlowHead(
            step_dim=step_dim, context_dim=shared_encoder.final_feature_dim,
            pred_len=pred_len, chunk_size=chunk_size, flow_layers=flow_layers, hidden_features=hidden_features, num_bins=num_bins
        )
        # Residual AR correction network
        self.ar_net = ResidualAR(context_dim=shared_encoder.final_feature_dim, pred_len=pred_len, enc_in=step_dim ,hidden=ar_hidden)

    def forward(self, context, y=None): # x 대신 context를 직접 받음
        if self.training and y is not None:
            B, L, D = y.shape
            # Compute AR predicted correction and correct residuals before flow
            # ar_pred = self.ar_net(context).view(B, L, D) 
            ar_pred = torch.zeros(B, L, D, device=y.device) 
            corrected = y - ar_pred  # we ask flow to model "corrected residual"
            log_prob = self.decoder.log_prob(corrected, context)
            # return both log_prob and ar_pred so caller can compute mse etc if needed
            return log_prob, ar_pred
        else:  # for sampling
            return context

    def sample(self, x, num_samples=1):
        # returns samples in original residual space (i.e. after adding ar_pred back)
        self.eval()
        with torch.no_grad():
            # 💡 [수정] self.forward(x) 대신 self.shared_encoder(x)를 직접 호출합니다.
            # 이렇게 하면 x (3D 텐서)가 인코더를 거쳐 2D 요약 컨텍스트로 변환됩니다.
            _, summary_context = self.shared_encoder(x)

            # 이제 decoder.sample에는 올바른 2D 컨텍스트가 전달됩니다.
            samples = self.decoder.sample(num_samples, summary_context)
            _, b, c, d = samples.shape

            ar_pred = self.ar_net(summary_context).view(b, c, d)

            # add back AR correction
            samples = samples + ar_pred # broadcast add
        return samples
