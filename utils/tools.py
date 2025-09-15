import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import csv, os

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.switch_backend('agg')

def dict_to_string(dict: dict):
    ret = ""
    for k, v in dict.items():
        ret += k + ": " + str(v.item()) +", "
    return ret

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# 내 모델 전용
def predict_and_visualize(deterministic_model, residual_model, detrender, scaler, x_test_raw, y_true_raw, seq_len, pred_len, title, num_samples=50, output_path="../pic/model_prediction_final_corrected.png"):
        """
        주어진 모델과 데이터로 예측을 수행하고 결과를 시각화합니다.

        Args:
            deterministic_model (nn.Module): 결정론적 예측 모델.
            residual_model (nn.Module): 확률론적 잔차 모델.
            detrender (nn.Module): 추세 제거/재결합 모듈.
            scaler (StandardScaler): 데이터 스케일러.
            x_test_raw (torch.Tensor): 원본 테스트 입력 데이터 (1, seq_len).
            y_true_raw (torch.Tensor): 원본 실제 미래 값 (pred_len,).
            seq_len (int): 입력 시퀀스 길이.
            pred_len (int): 예측 시퀀스 길이.
            num_samples (int): 잔차 샘플링 횟수.
            output_path (str): 결과 그래프를 저장할 경로.
        """
        deterministic_model.eval()
        residual_model.eval()
        
        # 올바른 순서로 전처리 적용
        x_test_detrended, future_trend = detrender(x_test_raw)
        x_test_scaled_np = scaler.transform(x_test_detrended.detach().numpy().reshape(-1, 1)).reshape(x_test_detrended.shape)
        x_test_scaled = torch.tensor(x_test_scaled_np, dtype=torch.float32)

        with torch.no_grad():
            # 모델 예측 (스케일링된 공간에서)
            mean_pred_scaled = deterministic_model(x_test_scaled)
            residual_samples_scaled1 = residual_model.sample(x_test_scaled, num_samples=1).squeeze(1)
            residual_samples_scaled2 = residual_model.sample(x_test_scaled, num_samples=num_samples).squeeze(1)
            
            # 스케일 복원
            final_samples_scaled1 = mean_pred_scaled.unsqueeze(0) + residual_samples_scaled1
            final_samples_scaled2 = mean_pred_scaled.unsqueeze(0) + residual_samples_scaled2
            final_samples_detrended_np1 = scaler.inverse_transform(final_samples_scaled1.detach().numpy().reshape(-1, 1)).reshape(1, pred_len)
            final_samples_detrended_np2 = scaler.inverse_transform(final_samples_scaled2.detach().numpy().reshape(-1, 1)).reshape(num_samples, pred_len)
            final_samples_detrended1 = torch.tensor(final_samples_detrended_np1, dtype=torch.float32)
            final_samples_detrended2 = torch.tensor(final_samples_detrended_np2, dtype=torch.float32)

            # 추세 재결합
            final_samples_raw1 = final_samples_detrended1 + future_trend
            final_samples_raw2 = final_samples_detrended2 + future_trend

        # 예측 결과 통계 계산
        mu_pred = final_samples_raw1.mean(dim=0)
        lower_bound = final_samples_raw2.kthvalue(int(0.05 * num_samples), dim=0).values
        upper_bound = final_samples_raw2.kthvalue(int(0.95 * num_samples), dim=0).values
        
        # 시각화
        plt.figure(figsize=(14, 7))
        plt.plot(range(seq_len), x_test_raw.flatten().detach().numpy(), label="Input Data", color='black')
        future_range = range(seq_len, seq_len + pred_len)
        plt.plot(future_range, y_true_raw.detach().numpy(), 'o-', label="Actual Future", color='green', markersize=3)
        plt.plot(future_range, mu_pred.detach().numpy(), 'o-', label="Predicted Mean", color='red', markersize=3)
        plt.fill_between(future_range, lower_bound.numpy(), upper_bound.numpy(), color='red', alpha=0.2, label="90% Confidence Interval")
        plt.title(title, fontsize=16)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(output_path)

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # get_model_complexity_info:  네트워크의 연산량과 패러미터 수를 출력
        # model: nn.Modules 클래스로 만들어진 객체. 연산량과 패러미터 수를 측정할 네트워크입니다.
        # input_res: 입력되는 텐서의 모양을 나타내는 tuple. 이 때, batch size에 해당하는 차원은 제외합니다.
        # print_per_layer_stat: True일 시, Layer 단위로 연산량과 패러미터 수를 출력합니다.
        # as_strings: True일 시, 연산량 및 패러미터 수를 string으로 변환하여 출력합니다.
        # verbose: True일 시, zero-op에 대한 warning을 출력합니다.

class SimpleLogger:
    def __init__(self, log_dir="logs", filename="train_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        # 헤더 작성
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stage","epoch","step",
                                 "det_mse","res_mse","nll",
                                 "grad_norm","lr","time"])

    def log(self, **kwargs):
        # 콘솔 출력
        msg = " | ".join([f"{k}:{v:.5f}" if isinstance(v,(float,int)) else f"{k}:{v}"
                          for k,v in kwargs.items()])
        print(msg)

        # CSV 기록
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([kwargs.get("stage"),
                             kwargs.get("epoch"),
                             kwargs.get("step"),
                             kwargs.get("det_mse", ""),
                             kwargs.get("res_mse", ""),
                             kwargs.get("nll", ""),
                             kwargs.get("grad_norm", ""),
                             kwargs.get("lr", ""),
                             kwargs.get("time","")])
