# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import time

import torch

print(torch.cuda.is_available())  # 查看GPu设备是否可用
print(torch.cuda.device_count())  # 查看GPu设备数量
print(torch.cuda.get_device_name())  # 查看当前GPu设备名称，默认设备id从0开始
print(torch.cuda.current_device())

# +
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import dotdict

fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = dotdict()
args.root_path = "./dataset/WISDM/"
args.data_path = "WISDM.txt"
args.model_id = "WISDM"
args.model = "AutoTimes_Gemma"
args.data = "WISDM"
args.features = "M"
args.seq_len = 200
args.token_len = 10
args.test_seq_len = 200
args.test_pred_len = 1
args.batch_size = 16
args.learning_rate = 0.0005
args.mlp_hidden_layers = 0
args.mlp_hidden_dim = 0
args.train_epochs = 4
args.use_amp = True
args.gpu = 0
args.cosine = True
args.tmax = 10
args.mix_embeds = False
args.drop_last = True
args.checkpoints = "./checkpoints/"
args.val_set_shuffle = True
args.dropout = 0.1
args.mlp_activation = "gelu"
args.num_workers = 10
args.patience = 1
args.des = "classification"
args.loss = "CrossEntropy"
args.lradj = "type1"
args.weight_decay = 0
args.warmup_epochs = 2  # Add warmup epochs for cosine scheduler
args.test_dir = "classification_WISDM_AutoTimes"
args.test_file_name = "checkpoint.pth"
args.visualize = False

print("Args in experiment:")
print(args)
# -

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
exp = Exp_Long_Term_Forecast(args)
# setting record of experiments
setting = "{}_{}_{}_sl{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}".format(
    args.model_id,
    args.model,
    args.data,
    args.seq_len,
    args.token_len,
    args.learning_rate,
    args.batch_size,
    args.weight_decay,
    args.mlp_hidden_dim,
    args.mlp_hidden_layers,
    args.cosine,
    args.mix_embeds,
    args.des,
)
print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
start_time = time.time()
exp.train(setting)
end_time = time.time()
training_time_seconds = end_time - start_time
print(f"Training completed in {training_time_seconds:.2f} seconds")
print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
exp.test(setting)
torch.cuda.empty_cache()
