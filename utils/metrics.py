import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CRPS(samples, y_true):
    """
    샘플 기반 CRPS Loss 계산
    Args:
        samples (Tensor): 모델 예측 샘플들 (num_samples, B, pred_len)
        y_true (Tensor): 실제 값 (B, pred_len)
    """
    # 예측 샘플들과 실제 값 사이의 평균 절대 오차
    term1 = torch.abs(samples - y_true.unsqueeze(0)).mean(dim=0)
    # 예측 샘플들끼리의 평균 절대 오차
    term2 = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)).mean(dim=[0, 1])
    
    loss = (term1 - 0.5 * term2).mean()
    return loss


def quantile_loss(y_pred, y_true, q):
    e = y_true - y_pred
    return torch.max((q-1)*e, q*e).mean()


def energy_score(y_true: torch.Tensor, y_pred_samples: torch.Tensor) -> torch.Tensor:
    """
    다변량 에너지 스코어를 계산합니다.

    Args:
        y_true (torch.Tensor): 실제 관측값 텐서. 
                               Shape: (batch_size, feature_dim)
        y_pred_samples (torch.Tensor): 예측 분포에서 샘플링한 텐서. 
                                       Shape: (batch_size, num_samples, feature_dim)

    Returns:
        torch.Tensor: 배치에 대한 평균 에너지 스코어 (스칼라 값).
    """
    batch_size, pred_len, feature_dim = y_pred_samples.shape
    y_true = y_true[:, :pred_len, :]

    # ------------------------------------------------------------------
    # Term 1: 예측 샘플과 실제 값 사이의 평균 거리
    # ||x_i - y||_2
    # ------------------------------------------------------------------
    
    # 각 샘플과 실제 값의 유클리드 거리(L2 norm) 계산
    # 결과 Shape: (batch_size, num_samples)
    term1_distances = torch.linalg.norm(y_pred_samples - y_true, ord=2, dim=-1)
    
    # 샘플에 대해 평균
    # 결과 Shape: (batch_size,)
    term1 = torch.mean(term1_distances, dim=1)

    # ------------------------------------------------------------------
    # Term 2: 예측 샘플들 사이의 평균 거리
    # 0.5 * ||x_i - x_j||_2
    # ------------------------------------------------------------------
    # 배치 내 각 항목에 대해 샘플들 간의 쌍별(pairwise) 유클리드 거리 계산
    # torch.cdist는 매우 효율적인 연산을 제공합니다.
    # 결과 Shape: (batch_size, num_samples, pred_len, pred_len)
    pairwise_distances = torch.cdist(y_pred_samples, y_pred_samples, p=2)
    
    # 모든 쌍별 거리에 대해 평균
    # 결과 Shape: (batch_size,)
    term2 = torch.mean(pairwise_distances, dim=(1, 2))

    # ------------------------------------------------------------------
    # 최종 에너지 스코어 계산
    # ------------------------------------------------------------------
    # ES = Term1 - 0.5 * Term2
    # 결과 Shape: (batch_size,)
    score_per_batch_item = term1 - 0.5 * term2
    
    # 전체 배치의 평균 스코어를 최종 손실로 반환
    return torch.mean(score_per_batch_item)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
