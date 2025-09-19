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
                    activation=F.relu,
                ),
                num_bins=num_bins, tails="linear", tail_bound=5.0
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
        
        # 각 패치를 d_model 차원의 벡터로 변환할 Linear 레이어
        # 패치는 (patch_len * c_in) 크기의 1D 벡터로 펼쳐집니다.
        self.projection = nn.Linear(patch_len * c_in, d_model)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        # 예: [512, 336, 7]
        
        # 1. 시퀀스를 패치 단위로 자릅니다. (stride 만큼 겹치면서)
        # unfold: 텐서를 슬라이딩 윈도우 방식으로 잘라주는 매우 효율적인 함수
        # (B, L, C) -> (B, C, L) -> unfold -> (B, C, Num_Patches, Patch_Len)
        x_unfolded = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 2. 패치를 펼치기(flatten) 위해 차원을 재정렬하고 합칩니다.
        # (B, C, Num_Patches, Patch_Len) -> (B, Num_Patches, C, Patch_Len)
        x_patched = x_unfolded.permute(0, 2, 1, 3)
        
        # (B, Num_Patches, C, Patch_Len) -> (B, Num_Patches, C * Patch_Len)
        B, n_patches, C, P = x_patched.shape
        x_flattened = x_patched.reshape(B, n_patches, -1)
        
        # 3. Linear 레이어를 통과시켜 각 패치를 d_model 차원의 벡터로 임베딩
        # (B, Num_Patches, C * Patch_Len) -> (B, Num_Patches, d_model)
        out = self.projection(x_flattened)
        
        return out