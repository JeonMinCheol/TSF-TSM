import torch

# 상삼각행렬 마스킹
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device) 

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

def random_segment_mask(x, mask_ratio=0.1):
    """
    Random Segment Masking
    x: [B, L, C]
    mask_ratio: 전체 길이 L 중 몇 %를 랜덤하게 가릴지
    """
    B, L, C = x.shape
    mask_len = max(1, int(L * mask_ratio))
    start = torch.randint(0, L - mask_len + 1, (1,)).item()
    
    x_masked = x.clone()
    x_masked[:, start:start+mask_len, :] = 0.0
    return x_masked
