import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv, os
import seaborn as sns

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
        
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, rank=0):
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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        os.makedirs(path, exist_ok=True)
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


def visual(true, preds=None, name='./pic/test.pdf',
           title="Forecast vs Ground Truth",
           xlabel="Time", ylabel="Value"):
    
    plt.figure(figsize=(10, 4))                       
    plt.plot(true, label='Ground Truth',
             color="#1e7cc0", linewidth=1.4, alpha=0.9)
    if preds is not None:
        plt.plot(preds, label='Prediction',
                 color="#ff0000", linewidth=1.0, alpha=0.7)

    # plt.title(title, fontsize=14, weight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 눈금/격자 가독성
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.legend(fontsize=11, loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.close()

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    
    # from ptflops import get_model_complexity_info    
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    #     print('Flops:' + flops)
    #     print('Params:' + params)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

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



def plot_attention_heatmap(attention_weights, layer_num, head_num, title, sample_num=0):
    """
    주어진 어텐션 가중치로 히트맵을 그리는 함수.

    Args:
        attention_weights (torch.Tensor): 어텐션 가중치 텐서. Shape: [B, n_heads, L, L]
        layer_num (int): 시각화할 레이어 번호.
        head_num (int): 시각화할 헤드 번호.
        sample_num (int): 시각화할 배치의 샘플 번호.
    """
    # 시각화할 특정 샘플, 특정 헤드의 어텐션 맵 선택
    # CPU로 데이터 이동 및 numpy 배열로 변환
    attn_map = attention_weights[sample_num, head_num].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_map, cmap='viridis') # 'viridis', 'plasma' 등 다양한 색상 맵 사용 가능
    
    plt.title(f'Attention Heatmap (Layer {layer_num+1}, Head {head_num+1})')
    plt.xlabel('Key (Attended Patches)')
    plt.ylabel('Query (Current Patches)')
    plt.savefig()


import io
from PIL import Image
import numpy as np

def get_heatmap_image_tensor(attention_weights, sample_num=0, head_num=0):
    """
    어텐션 가중치로 히트맵을 그려 PyTorch 이미지 텐서로 반환하는 함수.
    
    Args:
        attention_weights (torch.Tensor): 어텐션 가중치. Shape: [B, n_heads, L, L]
        sample_num (int): 시각화할 배치의 샘플 번호.
        head_num (int): 시각화할 헤드 번호.

    Returns:
        torch.Tensor: 이미지 텐서. Shape: [C, H, W]
    """
    # 특정 어텐션 맵 선택
    attn_map = attention_weights[sample_num, head_num].detach().cpu().numpy()

    # Matplotlib Figure 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn_map, cmap='viridis', ax=ax, cbar=False) # cbar는 생략 가능
    ax.set_title(f'Attention Heatmap (Head {head_num+1})')
    ax.set_xlabel('Key Patches')
    ax.set_ylabel('Query Patches')
    fig.tight_layout()

    # 💡 Figure를 메모리 버퍼에 PNG 이미지로 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # 💡 버퍼의 이미지를 PIL로 열고 Numpy 배열로 변환
    img = Image.open(buf)
    img_array = np.array(img.convert('RGB'))

    # 💡 Numpy 배열을 PyTorch 텐서로 변환하고, 채널 순서 변경 [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    # 메모리 누수 방지를 위해 figure와 buffer를 닫음
    plt.close(fig)
    buf.close()

    return img_tensor