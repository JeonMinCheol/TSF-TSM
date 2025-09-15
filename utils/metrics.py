import numpy as np


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

def crps_loss(samples, y_true):
    """
    샘플 기반 CRPS Loss 계산
    Args:
        samples (Tensor): 모델 예측 샘플들 (num_samples, B, pred_len)
        y_true (Tensor): 실제 값 (B, pred_len)
    """
    num_samples = samples.shape[0]
    # 예측 샘플들과 실제 값 사이의 평균 절대 오차
    term1 = torch.abs(samples - y_true.unsqueeze(0)).mean(dim=0)
    # 예측 샘플들끼리의 평균 절대 오차
    term2 = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)).mean(dim=[0, 1])
    
    loss = (term1 - 0.5 * term2).mean()
    return loss

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
