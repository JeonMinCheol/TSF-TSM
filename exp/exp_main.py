from ast import mod
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, TSF_TSM, iTransformer, ModernTCN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, plot_attention_heatmap, get_heatmap_image_tensor
from utils.metrics import metric
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import torch.distributed as dist

import random
import os
import time

import warnings
import matplotlib.pyplot as plt

import logging

from torch.utils.tensorboard import SummaryWriter

from torch.profiler import profile, record_function, ProfilerActivity
torch.backends.cudnn.deterministic=True
warnings.filterwarnings('ignore')
    
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # --- DDP 초기화 ---
        if self.args.use_multi_gpu and self.args.use_gpu:
            dist.init_process_group(
                backend="nccl",      # GPU라면 NCCL이 가장 빠름
                init_method="env://",
            )
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.rank = dist.get_rank()
        else:
            # 단일 GPU/CPU
            self.device = torch.device("cuda" if self.args.use_gpu else "cpu")
            self.rank = 0

        # rank 0만 로그 작성
        if self.rank == 0:
            log_dir = f"/data/a2019102224/PatchTST_supervised/tensor_logs/{self.args.model_id}_{self.args.model}/"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.writer.add_scalar("scalar/stride", self.args.stride)
            self.writer.add_scalar("scalar/window_size", self.args.moving_avg)

        # --- 모델 생성 ---
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'iTransformer': iTransformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'TSF_TSM': TSF_TSM,
            'ModernTCN': ModernTCN,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        # --- DDP 래핑 ---
        if self.args.use_multi_gpu and self.args.use_gpu:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.device)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index],   # 또는 [local_rank]
                output_device=self.device.index,
                find_unused_parameters=True
            )
            if self.rank == 0:
                print(f"[DDP] rank {self.rank}, local_rank {local_rank} -> device {self.device}")
        else:
            model = model.to(self.device)

        return model


    def _get_data(self, flag):
        data_set, data_loader, sampler = data_provider(self.args, flag)
        return data_set, data_loader, sampler

    def _select_optimizer(self):
        self.model = self.model.module if self.args.use_multi_gpu and self.args.use_gpu else self.model
        
        if self.args.model == "TSF_TSM":
            model_optim = [
                optim.Adam(list(self.model.adaptive_norm_block.parameters()) + list(self.model.mean_head.parameters()), lr=self.args.learning_rate),
                optim.Adam(self.model.residual_head.parameters(), lr=self.args.learning_rate)
            ]
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc="Validation")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "TSF_TSM":
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.sample(batch_x)
                    else:
                        outputs = self.model.sample(batch_x)

                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        # --- 💡 텐서보드 어텐션 시각화 로깅 (vali 루프 끝난 후) ---
        # 검증 데이터로더에서 첫 번째 배치만 가져와서 시각화

        if 'TSF_TSM' in self.args.model:
            vis_batch_x, _, _, _ = next(iter(vali_loader))
            vis_batch_x = vis_batch_x.float().to(self.device)

            # 모델을 통해 어텐션 가중치 추출
            normalized_x, _, _, _ = self.model.adaptive_norm_block.normalize(vis_batch_x)
            _, all_attention_weights = self.model.encoder(normalized_x, get_attn=True)

            # 각 레이어와 헤드별로 히트맵 이미지를 텐서보드에 기록
            for layer_idx, attn_weights in enumerate(all_attention_weights):
                # 예시로 첫 4개의 헤드만 기록
                num_heads_to_log = min(4, attn_weights.shape[1]) 
                for head_idx in range(num_heads_to_log):
                    # 히트맵 이미지 텐서 생성
                    heatmap_tensor = get_heatmap_image_tensor(attn_weights, head_num=head_idx)
                    
                    # 텐서보드에 이미지 추가
                    if self.rank == 0:
                        self.writer.add_image(
                            tag=f'Attention/Layer_{layer_idx+1}/Head_{head_idx+1}', 
                            img_tensor=heatmap_tensor, 
                            global_step=epoch # 현재 에폭 번호를 기록
                        )

        total_loss = np.average(total_loss)
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader, sampler = self._get_data(flag='train')
        vali_data, vali_loader, _ = self._get_data(flag='val')
        test_data, test_loader, _ = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        if self.args.model == "TSF_TSM":
            deter_optim, residual_opitm = model_optim
            deter_scheduler = lr_scheduler.OneCycleLR(optimizer = deter_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
            residual_scheduler = lr_scheduler.OneCycleLR(optimizer = residual_opitm,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            train_loss, deter_train_loss, residual_train_loss = [], [], []
            iter_count = 0

            self.model.train()
            sampler.set_epoch(epoch) if sampler is not None else None
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model != "TSF_TSM":
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x, batch_y)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            # 모델 forward 한 번 호출
                            deter_optim.zero_grad()
                            residual_opitm.zero_grad()

                            deter_loss, residual_loss = self.model(batch_x, batch_y)
                            total_loss = (1 - self.args.alpha) * deter_loss + self.args.alpha * residual_loss

                            scaler.scale(total_loss).backward()
                            scaler.step(deter_optim)
                            scaler.step(residual_opitm)
                            scaler.update()

                        deter_train_loss.append(deter_loss.item())
                        residual_train_loss.append(residual_loss.item())
                    else:
                        # if epoch == 0 and i == 5:
                        #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                        #         with record_function("model_inference"): # 이 블록에 'model_inference'라는 이름을 붙임
                        #             deter_optim.zero_grad()
                        #             residual_opitm.zero_grad()

                        #             deter_loss, residual_loss = self.model(batch_x, batch_y)
                        #             total_loss = deter_loss + self.args.alpha * residual_loss

                        #             total_loss.backward()
                        #             deter_optim.step()
                        #             residual_opitm.step()

                        #             deter_train_loss.append(deter_loss.item())
                        #             residual_train_loss.append(residual_loss.item())

                        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                        deter_optim.zero_grad()
                        residual_opitm.zero_grad()

                        deter_loss, residual_loss = self.model(batch_x, batch_y)
                        total_loss = deter_loss + self.args.alpha * residual_loss

                        total_loss.backward()
                        deter_optim.step()
                        residual_opitm.step()

                        deter_train_loss.append(deter_loss.item())
                        residual_train_loss.append(residual_loss.item())

                if self.args.model == "TSF_TSM" and self.rank == 0:
                    self.writer.add_scalar("residual/mean", self.model.residual_mean, epoch)
                    self.writer.add_scalar("residual/std", self.model.residual_std, epoch)

                    self.writer.add_scalar("train/Model Optimizer LR", deter_optim.param_groups[0]['lr'], epoch)
                    self.writer.add_scalar("train/Deterministic Loss", deter_loss, epoch)
                    self.writer.add_scalar("train/Residual Loss", residual_loss, epoch)
                    self.writer.add_scalar("train/Total Loss", total_loss, epoch)
                    self.writer.flush()

                elif self.rank == 0:
                    self.writer.add_scalar("train/Model Optimizer LR", model_optim.param_groups[0]['lr'], epoch)
                    self.writer.add_scalar("train/Total Loss", loss.item(), epoch)
                    self.writer.flush()

                if (i + 1) % 100 == 0:
                    if self.args.model != "TSF_TSM":
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | Deter loss: {2:.7f}, Residual loss: {3:.7f}".format(i + 1, epoch + 1, deter_loss.item(), residual_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj == 'TST':
                    if self.args.model == "TSF_TSM":
                        # TSF_TSM 모델은 두 스케줄러를 모두 업데이트
                        deter_scheduler.step()
                        residual_scheduler.step()
                    else:
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
                        scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)

            if self.args.model == "TSF_TSM":
                deter_train_loss = np.average(deter_train_loss)
                residual_train_loss = np.average(residual_train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)
            test_loss = self.vali(test_data, test_loader, criterion, epoch)

            if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                self.writer.add_scalar("vali/loss", vali_loss, epoch)
                self.writer.add_scalar("test/loss", test_loss, epoch)
                self.writer.flush()

                if self.args.model != "TSF_TSM":
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                else:
                    print("Epoch: {0}, Steps: {1} | Deter Train Loss: {2:.7f} Residual Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                        epoch + 1, train_steps, deter_train_loss, residual_train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: 
                print("Early stopping")

            if self.args.lradj != 'TST':
                if self.args.model == "TSF_TSM":
                    adjust_learning_rate(deter_optim, deter_scheduler, epoch + 1, self.args, True)
                    adjust_learning_rate(residual_opitm, residual_scheduler, epoch + 1, self.args, True)
                else:
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, True)
            else:
                if self.args.model == "TSF_TSM":
                    if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                        print('Updating Deterministic learning rate to {}'.format(deter_scheduler.get_last_lr()[0]))
                        print('Updating Residual learning rate to {}'.format(residual_scheduler.get_last_lr()[0]))
                else:
                    if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
                        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.model.eval()
        test_data, test_loader, _ = self._get_data(flag='test')
        
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print('loading model')

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Test")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "TSF_TSM":
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.sample(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                    else:
                        outputs = self.model.sample(batch_x)

                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0: # 숫자 변경
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, corr:{}'.format(mae, mse, rmse, mape, mspe, rse, corr))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, corr:{}'.format(mae, mse, rmse, mape, mspe, rse, corr))
            f.write('\n')
            f.write('\n')
            f.close()
            np.save(folder_path + 'pred.npy', preds)

            self.writer.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(pred_loader, desc="Predict")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == "TSF_TSM":
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.sample(batch_x)
                    else:
                        outputs = self.model.sample(batch_x)
                        
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            elif 'TCN' in self.args.model:
                                outputs = self.model(batch_x, batch_x_mark)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        if (self.args.use_multi_gpu and self.rank == 0) or not self.args.use_multi_gpu:
            np.save(folder_path + 'real_prediction.npy', preds)

        return
