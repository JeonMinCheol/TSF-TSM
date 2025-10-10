import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.rank = int(os.environ.get("RANK", 0)) if self.args.use_multi_gpu and self.args.use_gpu else 0
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device('cuda:{}'.format(local_rank))
            print('Use GPU: cuda:{}'.format(local_rank))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
