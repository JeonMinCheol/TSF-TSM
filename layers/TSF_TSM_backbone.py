import torch
import torch.nn as nn
from .TSF_TSM_layers import *
from .TSF_TSM_experts_blocks import APGELU
import logging

class MeanPredictionHead(nn.Module):
    """
    메모리 효율적인 MLP 기반의 결정론적 예측 헤드.
    """
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        d_model = configs.d_model
        hidden_dim = d_model * 8

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            APGELU(hidden_dim),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            APGELU(hidden_dim),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim, self.pred_len * self.enc_in)
        )

    def forward(self, summary_context):
        # summary_context shape: [B, d_model]
        output = self.head(summary_context) # [B, pred_len * enc_in]
        
        # 최종 shape으로 재구성: [B, pred_len, enc_in]
        output = output.view(-1, self.pred_len, self.enc_in)
        return output

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
        y_windows_flat = y_windows_flat + 0.003 * torch.randn_like(y_windows_flat)  # 작은 노이즈 추가
        
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
            
            final_samples = final_samples / counts.unsqueeze(0).clamp(min=1.0)

        return final_samples