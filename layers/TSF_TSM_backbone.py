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
from .TSF_TSM_layers import *

# nflows 라이브러리 임포트
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.normalization import BatchNorm
from nflows.nn.nets import ResidualNet

import logging

class PredictionHead(nn.Module):
    """
    인코더가 만든 요약 컨텍스트 벡터를 받아, 
    미래 예측 시퀀스를 한 번에 출력하는 간단한 선형 헤드.
    """
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.head = nn.Linear(configs.d_model, configs.pred_len * configs.enc_in)

    def forward(self, summary_context):
        # summary_context shape: [B, d_model]
        output_flat = self.head(summary_context)
        # output shape: [B, pred_len, c_in]
        return output_flat.view(-1, self.pred_len, self.enc_in)

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

class ProbabilisticResidualModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.c_in = configs.enc_in
        self.pred_len = configs.pred_len
        context_dim = configs.d_model
        
        self.window_size = configs.moving_avg
        self.stride = configs.stride

        feature_dim = self.c_in * self.window_size
        
        self.flow_head = create_conditional_nsf_flow(
            feature_dim=feature_dim, 
            context_dim=context_dim,
            num_layers=configs.flow_layers, 
            hidden_features=configs.hidden_features, 
            num_bins=configs.num_bins
        )

    def forward(self, summary_context, y):
        # y shape: [B, pred_len, c_in]
        B, L, C = y.shape
        
        # --- 💡 [핵심] for 루프를 unfold 연산으로 대체 ---
        # 1. y를 [B, C, L] 형태로 변환
        y_transposed = y.permute(0, 2, 1)
        
        # 2. unfold를 사용하여 모든 윈도우를 한 번에 추출
        # 결과 shape: [B, C, num_windows, window_size]
        y_windows = y_transposed.unfold(dimension=2, size=self.window_size, step=self.stride)
        
        # 3. Flow에 입력하기 위해 shape 재정렬 및 flatten
        # [B, C, num_windows, window_size] -> [B, num_windows, C, window_size]
        y_windows = y_windows.permute(0, 2, 1, 3)
        num_windows = y_windows.shape[1]
        
        # [B, num_windows, C, window_size] -> [B * num_windows, C * window_size]
        y_windows_flat = y_windows.reshape(B * num_windows, -1)
        
        # 4. summary_context를 각 윈도우에 맞게 확장
        # [B, d_model] -> [B, num_windows, d_model] -> [B * num_windows, d_model]
        expanded_context = summary_context.unsqueeze(1).expand(-1, num_windows, -1)
        expanded_context = expanded_context.reshape(B * num_windows, -1)
        
        # 5. 모든 윈도우에 대해 Flow를 병렬로 한 번에 실행
        log_prob_windows = self.flow_head.log_prob(y_windows_flat, context=expanded_context)
        
        # 6. 전체 평균 Loss 계산
        total_log_prob = log_prob_windows.mean()
        
        return total_log_prob

    def sample(self, summary_context, num_samples=1):
        self.eval()
        with torch.no_grad():
            B = summary_context.size(0)
            num_windows = (self.pred_len - self.window_size) // self.stride + 1
            
            # 1. 컨텍스트를 모든 윈도우에 맞게 확장
            expanded_context = summary_context.unsqueeze(1).expand(-1, num_windows, -1)
            expanded_context = expanded_context.reshape(B * num_windows, -1)
            
            # 2. 모든 윈도우에 대한 샘플을 병렬로 한 번에 생성
            # sample_windows_flat shape: [num_samples, B * num_windows, window_size * c_in]
            sample_windows_flat = self.flow_head.sample(num_samples, context=expanded_context)
            
            # 3. 윈도우 shape으로 복원
            # [num_samples, B, num_windows, window_size, c_in]
            sample_windows = sample_windows_flat.view(num_samples, B, num_windows, self.window_size, self.c_in)
            
            # 4. 겹치는(Overlapping) 윈도우들을 평균내어 최종 시퀀스 생성 (Overlap-add 방식)
            final_samples = torch.zeros(num_samples, B, self.pred_len, self.c_in, device=summary_context.device)
            counts = torch.zeros(B, self.pred_len, self.c_in, device=summary_context.device)
            
            for i in range(num_windows):
                start_idx = i * self.stride
                end_idx = start_idx + self.window_size
                final_samples[:, :, start_idx:end_idx, :] += sample_windows[:, :, i, :, :]
                counts[:, start_idx:end_idx, :] += 1
            
            final_samples = final_samples / counts.unsqueeze(0)

        return final_samples

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # 디코더의 입력(타겟 시퀀스)을 d_model 차원으로 변환하는 임베딩 레이어
        self.embedding = nn.Linear(configs.enc_in, configs.d_model)
        
        # DecoderLayer를 num_layers 만큼 쌓음
        self.layers = nn.ModuleList([
            DecoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout) for _ in range(configs.num_layers)
        ])
        
        # 최종 출력을 원래 피처 차원(c_in)으로 변환하는 프로젝션 레이어
        self.projection = nn.Linear(configs.d_model, configs.enc_in)

    def forward(self, x, memory):
        # x: 디코더의 입력. shape: [Batch, Target_Seq_Len, c_in]
        # memory: 인코더의 전체 출력. shape: [Batch, Source_Seq_Len, d_model]
        
        # 1. 입력 임베딩
        x = self.embedding(x)
        
        # 2. Self-Attention을 위한 Look-ahead 마스크 생성
        #    디코더가 예측 시점 t에서, t보다 미래의 정보를 보지 못하게 함
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # 3. 모든 디코더 레이어를 순차적으로 통과
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
            
        # 4. 최종 출력
        return self.projection(x)