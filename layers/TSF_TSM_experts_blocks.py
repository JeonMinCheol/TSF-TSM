import torch
import torch.nn as nn
import torch.nn.functional as F
from .TSF_TSM_layers import PatchingEmbedding
from .SelfAttention_Family import AttentionPool

class MoE(nn.Module):
    """Mixture of Experts FFN 레이어."""
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: [B, L, C]
        logits = self.gating_network(x) # [B, L, num_experts]
        gates = F.softmax(logits, dim=-1)
        
        # 각 토큰에 대해 top-k 전문가 선택
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1) # [B, L, top_k]
        
        # top-k 가중치 정규화
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # 전문가 계산
        output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_indices = top_k_indices.view(-1, self.top_k)

        for i in range(self.num_experts):
            # i번째 전문가를 선택한 토큰들의 마스크 생성
            mask, _ = (flat_indices == i).max(dim=-1)
            if mask.any():
                expert_input = flat_x[mask]
                expert_output = self.experts[i](expert_input)
                
                # 가중합을 위한 가중치 추출
                gate_values = top_k_gates.view(-1, self.top_k)[mask]
                indices_values = top_k_indices.view(-1, self.top_k)[mask]
                expert_gate = gate_values[indices_values == i]
                
                # 가중치를 적용하여 결과 저장
                output.view(-1, x.size(-1))[mask] += expert_output * expert_gate.unsqueeze(-1)
        return output

class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.moe_ffn = MoE(d_model, d_ff, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.moe_ffn(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class SharedEncoderWithMoE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patching_embedding = PatchingEmbedding(configs.patch_len, configs.stride, configs.enc_in, configs.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, (configs.seq_len - configs.patch_len) // configs.stride + 1, configs.d_model))
        self.encoder_layers = nn.ModuleList([
            MoETransformerEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.num_experts)
            for _ in range(configs.e_layers)
        ])
        self.attention_pool = AttentionPool(d_model=configs.d_model, n_heads=configs.n_heads)

    def forward(self, x):
        x_patched = self.patching_embedding(x)
        x_patched += self.pos_encoder
        
        for layer in self.encoder_layers:
            x_patched = layer(x_patched)
        
        summary_context = self.attention_pool(x_patched)
        return summary_context