import torch
import torch.nn as nn
import torch.nn.functional as F
from .TSF_TSM_layers import PatchingEmbedding
from .SelfAttention_Family import AttentionPool
import math

class ContextualCouplingTSA(nn.Module):
    """
    Affine Coupling 구조를 컨텍스트 인코딩에 적용한 TSA.
    
    입력의 절반(x1)으로 어텐션 컨텍스트를 만들고, 
    이를 이용해 다른 절반(x2)을 변형시킬 파라미터(s, t)를 생성합니다.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # 아핀 커플링을 위해 d_model은 짝수여야 합니다.
        assert d_model % 2 == 0, "d_model must be an even number."
        
        self.d_model = d_model
        self.d_channel = d_model // 2 # 입력을 둘로 나누므로 실제 처리 채널 수는 절반
        self.n_heads = n_heads
        self.d_head = self.d_channel // n_heads

        # 💡 변경 포인트: Q, K는 d_channel을 기준으로 생성 (V는 사용 안 함)
        self.query_projection = nn.Linear(self.d_channel, self.d_channel)
        self.key_projection = nn.Linear(self.d_channel, self.d_channel)
        
        # Saliency 계산을 위한 Conv1d (입력 채널도 절반)
        self.saliency_conv = nn.Conv1d(
            in_channels=self.d_channel,
            out_channels=n_heads,
            kernel_size=3,
            padding='same'
        )

        # 💡 핵심 포인트: 어텐션 컨텍스트로부터 scale(s)과 shift(t)를 생성하는 레이어
        # s와 t 모두 d_channel 크기를 가지므로 출력은 d_channel * 2
        self.st_projection = nn.Linear(self.d_channel, self.d_channel)
        
        # 최종 출력을 위한 선형 레이어 (원본 TSA와 동일)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None):
        # src shape: [B, L, d_model]
        
        # 1. 💡 입력을 x1, x2로 분할
        x1, x2 = src.chunk(2, dim=-1) # [B, L, d_channel]
        
        B, L, _ = x1.shape

        # 2. x1을 이용해 Q, K, Saliency 생성 (원본 TSA와 거의 동일)
        Q = self.query_projection(x1).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.key_projection(x1).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        saliency_scores = self.saliency_conv(x1.transpose(1, 2)).unsqueeze(2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention_scores = attention_scores + saliency_scores

        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)

        # 3. 💡 어텐션 가중치를 이용해 컨텍스트 벡터 생성 (V 대신 K 사용)
        # 인코더이므로 softmax를 사용하여 가중치를 만드는 것이 자연스럽습니다.
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # K를 가중합하여 컨텍스트 벡터를 만듭니다.
        context = torch.matmul(attention_weights, K) # [B, n_heads, L, d_head]
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_channel)

        # 4. 💡 컨텍스트 벡터로부터 게이트 파라미터 예측
        g_params = self.st_projection(context) 
        gate = torch.sigmoid(g_params)
        
        # 5. x2에 게이트 변환 적용
        transformed_x2 = torch.tanh(x2) 
        y2 = gate * transformed_x2 + (1 - gate) * x2

        # 6. 💡 변환되지 않은 x1과 변환된 y2를 합치고 최종 출력
        output = torch.cat([x1, y2], dim=-1) # [B, L, d_model]
        output = self.out_projection(output)

        return output, attention_weights # MoE 레이어와 호환되는 출력 형태

class TSA(nn.Module):
    """
    Temporal Saliency Attention (TSA)
    시계열의 각 시점(패치)별 중요도를 동적으로 학습하여 어텐션 스코어에 반영합니다.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q, K, V를 위한 선형 레이어
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 최종 출력을 위한 선형 레이어
        self.out_projection = nn.Linear(d_model, d_model)
        
        # 시간적 중요도(Saliency)를 학습하기 위한 1D Conv 레이어
        self.saliency_conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=n_heads, # 각 헤드별로 다른 saliency를 학습
            kernel_size=3, 
            padding='same'
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None):
        # src shape: [Batch, Seq_Len, d_model]
        B, L, _ = src.shape

        # 1. Q, K, V 생성 및 n_heads로 분할
        Q = self.query_projection(src).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.key_projection(src).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.value_projection(src).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        # Q, K, V shape: [B, n_heads, Seq_Len, d_head]

        # 2. 시간적 중요도(Saliency) 스코어 계산
        # src: [B, L, C] -> [B, C, L] for Conv1d
        saliency_scores = self.saliency_conv(src.transpose(1, 2)) # -> [B, n_heads, L]
        # 어텐션 스코어와 덧셈을 위해 차원 확장: [B, n_heads, 1, L]
        saliency_scores = saliency_scores.unsqueeze(2)

        # 3. 어텐션 스코어 계산 및 Saliency 반영
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 💡 핵심: Softmax 이전에 Saliency 스코어를 더해줌
        # Saliency 스코어가 브로드캐스팅되어 모든 Query에 대한 Key의 중요도로 더해짐
        attention_scores = attention_scores + saliency_scores

        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)

        # 4. 어텐션 가중치 계산 및 V와 결합
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V) # [B, n_heads, L, d_head]

        # 5. 헤드들을 다시 합치고 최종 출력
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.out_projection(output)

        return output, attention_weights # 어텐션 가중치도 반환하여 분석에 활용 가능    

class APGELU(nn.Module):
    """
    Adaptive Phase GELU (AP-GELU) - 수정된 버전
    - 2D 입력 [Tokens, Features]를 처리하도록 수정
    """
    def __init__(self, feature_dim): # d_model 대신 feature_dim으로 이름 변경
        super().__init__()
        self.omega_head = nn.Linear(feature_dim, feature_dim)
        self.phi_head = nn.Linear(feature_dim, feature_dim)

        nn.init.constant_(self.omega_head.weight, 0)
        nn.init.constant_(self.omega_head.bias, 1)
        nn.init.constant_(self.phi_head.weight, 0)
        nn.init.constant_(self.phi_head.bias, 0)

    def forward(self, x):
        # x shape: [Tokens, Features] (예: [4967, 256])
        
        # 💡 핵심: 토큰 차원(dim=0)에 대해 평균을 내어 통계치 계산
        # 이렇게 하면 현재 전문가에게 들어온 모든 토큰들의 평균적 특성을 반영
        if x.dim() < 2 or x.shape[0] <= 1: # 토큰이 하나이거나 없을 경우
             return F.gelu(x)

        token_stats = x.mean(dim=0) # -> [Features]

        # 통계치를 기반으로 omega와 phi 예측
        omega = F.softplus(self.omega_head(token_stats)) # -> [Features]
        phi = self.phi_head(token_stats) # -> [Features]
        
        # omega와 phi가 [Features] 형태이므로 [Tokens, Features] 형태인 x에 브로드캐스팅되어 연산됨
        transformed_x = omega * x + phi
        
        return F.gelu(transformed_x)

class MoE(nn.Module):
    """Mixture of Experts FFN 레이어."""
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                APGELU(d_ff),
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

class MoETSAEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts, dropout=0.1):
        super().__init__()
        # self.self_attn = TSA(d_model, n_heads, dropout=dropout)
        self.self_attn = ContextualCouplingTSA(d_model, n_heads, dropout=dropout)
        self.moe_ffn = MoE(d_model, d_ff, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None):
        src2, attn_weights = self.self_attn(src, attn_mask=attn_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.moe_ffn(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn_weights

class ContextEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patching_embedding = PatchingEmbedding(configs.patch_len, configs.stride, configs.enc_in, configs.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, (configs.seq_len - configs.patch_len) // configs.stride + 1, configs.d_model))
        self.encoder_layers = nn.ModuleList([
            MoETSAEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.num_experts, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        self.attention_pool = AttentionPool(d_model=configs.d_model, n_heads=configs.n_heads)

    def forward(self, x, get_attn=False):
        x_patched = self.patching_embedding(x)
        x_patched += self.pos_encoder

        attn_weights_list = []
        
        for layer in self.encoder_layers:
            x_patched, attn_weights  = layer(x_patched)
            if get_attn:
                attn_weights_list.append(attn_weights)
        
        summary_context = self.attention_pool(x_patched)

        if get_attn:
            return summary_context, attn_weights_list
        
        return summary_context