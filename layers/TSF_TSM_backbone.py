import torch
import math
import torch.nn as nn
from .TSF_TSM_layers import *
import logging

class MeanPredictionHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        d_model = configs.d_model
        d_ff = configs.d_ff

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.Dropout(configs.dropout),
            nn.Linear(d_ff, self.pred_len * self.enc_in)
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
        
        self.window_size = configs.patch_len
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
        nll_loss = -log_prob_windows.mean()
        
        return nll_loss

    def sample(self, summary_context, num_samples=5):
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

class GaussianResidualModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model
        c_in = configs.enc_in

        # context → [mean, log_var]
        self.fc = nn.Linear(d_model, 2 * c_in)

    def forward(self, context, y=None):
        params = self.fc(context)  # [B, 2*C]
        mean, log_var = params.chunk(2, dim=-1)

        # 분산 범위 확장
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        std = torch.exp(0.5 * log_var)

        if y is not None:
            B, L, C = y.shape
            mean = mean.unsqueeze(1).expand(B, L, C)
            std = std.unsqueeze(1).expand(B, L, C)
            log_var = log_var.unsqueeze(1).expand(B, L, C)

            # Gaussian NLL (안정화 버전)
            nll = 0.5 * (
                torch.clamp(log_var, min=-10.0, max=10.0) +
                ((y - mean) ** 2) / (torch.clamp(std**2, min=1e-6)) +
                math.log(2 * math.pi)
            )
            return nll.mean()
        else:
            return mean, std

    def sample(self, context, num_samples=1, length=1):
        """
        context: [B, d_model]
        return: [num_samples, B, length, C]
        """
        params = self.fc(context)
        mean, log_var = params.chunk(2, dim=-1)
        # log_var = torch.clamp(log_var, min=-5.0, max=2.0)
        std = torch.exp(0.5 * log_var)

        B, C = mean.shape
        mean = mean.unsqueeze(1).unsqueeze(0).expand(num_samples, B, length, C)
        std = std.unsqueeze(1).unsqueeze(0).expand(num_samples, B, length, C)

        eps = torch.randn_like(std)
        samples = mean + eps * std
        return samples