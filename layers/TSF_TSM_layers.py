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
        # 💡 각 변수(c_in)를 d_model로 독립적으로 투영합니다.
        self.projection = nn.Linear(patch_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model) # 안정성을 위해 추가
        # 💡 d_model * c_in 차원을 최종 d_model로 다시 투영할 레이어
        self.final_projection = nn.Linear(d_model * c_in, d_model)

    def forward(self, x):
        # x shape: [B, L, C] (C = c_in)
        B, L, C = x.shape

        # 1. 패치 생성
        x_unfolded = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_patched = x_unfolded.permute(0, 2, 1, 3) # [B, Num_Patches, C, Patch_Len]
        B, n_patches, C, P = x_patched.shape

        # 2. 💡 각 변수별로 독립적인 임베딩 수행
        # [B, n_patches, C, P] -> [B * n_patches * C, P]
        x_patched_flat = x_patched.reshape(-1, P)
        projected = self.projection(x_patched_flat) # [B * n_patches * C, d_model]

        # 3. 💡 변수들의 임베딩을 합치고 최종 차원으로 투영
        projected = projected.reshape(B, n_patches, C, -1) # [B, n_patches, C, d_model]
        projected_flat = projected.reshape(B, n_patches, -1) # [B, n_patches, C * d_model]

        out = self.final_projection(projected_flat) # [B, n_patches, d_model]

        return out