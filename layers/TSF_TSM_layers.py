import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# nflows 라이브러리 임포트
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
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
                    activation=F.leaky_relu,
                ),
                num_bins=num_bins, 
                tails="linear", 
                tail_bound=3.0, 
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
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
        
        # ✅ [변경] 패치 내의 모든 변수와 시점을 한번에 d_model로 투영합니다.
        # 입력 차원: 변수 개수(c_in) * 패치 길이(patch_len)
        self.projection = nn.Linear(c_in * patch_len, d_model)
        # LayerNorm은 안정적인 학습을 위해 유지하는 것이 좋습니다.
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, C]
        
        # 1. 패치 생성 (PyTorch의 unfold 함수 사용)
        # 결과 shape: [B, C, Num_Patches, Patch_Len]
        x_unfolded = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 2. ✅ [변경] 패치를 flatten하여 한번에 투영
        # [B, C, Num_Patches, Patch_Len] -> [B, Num_Patches, C, Patch_Len]
        x_patched = x_unfolded.permute(0, 2, 1, 3)
        
        # [B, Num_Patches, C, Patch_Len] -> [B, Num_Patches, C * Patch_Len]
        # 각 패치에 있는 모든 값들을 하나의 벡터로 만듭니다.
        x_patched = x_patched.reshape(x_patched.shape[0], x_patched.shape[1], -1)
        
        # [B, Num_Patches, C * Patch_Len] -> [B, Num_Patches, d_model]
        out = self.projection(x_patched)
        out = self.layer_norm(out) # LayerNorm 적용
        
        return out