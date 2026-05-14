import math
import os
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device("cuda:{}".format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (
                    self.args.use_multi_gpu and self.args.local_rank == 0
                ) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.AdamW(
            [{"params": p_list}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        if (
            self.args.use_multi_gpu and self.args.local_rank == 0
        ) or not self.args.use_multi_gpu:
            print("next learning rate is {}".format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                batch_y = batch_y.squeeze(-1)
                batch_y = torch.clamp(batch_y, 0, 5)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)

                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (
                        self.args.use_multi_gpu and self.args.local_rank == 0
                    ) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print(
                            "\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
                                i + 1, speed, left_time
                            )
                        )
                        iter_count = 0
                        time_now = time.time()
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(
                self.device
            )
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        print(len(train_loader))
        print(len(vali_loader))
        print(len(test_loader))

        path = os.path.join(self.args.checkpoints, setting)
        if (
            self.args.use_multi_gpu and self.args.local_rank == 0
        ) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()

        # Initialize scheduler with warmup
        if self.args.cosine:
            # Start with warmup
            warmup_epochs = getattr(self.args, "warmup_epochs", 2)
            total_epochs = self.args.train_epochs

            # Create a custom warmup + cosine annealing scheduler
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup
                    return (epoch + 1) / warmup_epochs
                else:
                    # Cosine annealing after warmup
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                model_optim, lr_lambda=lr_lambda
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                model_optim, T_0=self.args.tmax, T_mult=1, eta_min=1e-8
            )

        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.0, device="cuda")
            count = torch.tensor(0.0, device="cuda")

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                batch_y = batch_y.squeeze(-1)
                batch_y = torch.clamp(batch_y, 0, 5)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                        loss = criterion(outputs, batch_y)
                        loss_val += loss.item()
                        count += 1
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    loss = criterion(outputs, batch_y)
                    loss_val += loss.item()
                    count += 1

                if (i + 1) % 100 == 0:
                    if (
                        self.args.use_multi_gpu and self.args.local_rank == 0
                    ) or not self.args.use_multi_gpu:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                                i + 1, epoch + 1, loss.item()
                            )
                        )
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (
                            (self.args.train_epochs - epoch) * train_steps - i
                        )
                        print(
                            "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                                speed, left_time
                            )
                        )
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            if (
                self.args.use_multi_gpu and self.args.local_rank == 0
            ) or not self.args.use_multi_gpu:
                print(
                    "Epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    )
                )
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (
                self.args.use_multi_gpu and self.args.local_rank == 0
            ) or not self.args.use_multi_gpu:
                print(
                    "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                        epoch + 1,
                        train_steps,
                        train_loss,
                        vali_loss,  # , test_loss
                    )
                )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (
                    self.args.use_multi_gpu and self.args.local_rank == 0
                ) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (
                    self.args.use_multi_gpu and self.args.local_rank == 0
                ) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        print("info:", self.args.test_seq_len)
        if test:
            print("loading model")
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print(
                "loading model from {}".format(
                    os.path.join(self.args.checkpoints, setting, best_model_path)
                )
            )
            load_item = torch.load(
                os.path.join(self.args.checkpoints, setting, best_model_path)
            )
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in load_item.items()},
                strict=False,
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                batch_y = batch_y.squeeze(-1)
                batch_y = torch.clamp(batch_y, 0, 5)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)

                batch_y = batch_y.to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (
                        self.args.use_multi_gpu and self.args.local_rank == 0
                    ) or not self.args.use_multi_gpu:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print(
                            "\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
                                i + 1, speed, left_time
                            )
                        )
                        iter_count = 0
                        time_now = time.time()

                if self.args.visualize:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f"{self.args.test_pred_len}/"
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(gt, pd, os.path.join(dir_path, f"{i}.png"))

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()

        # Use classification metrics
        acc, f1, recall, precision, cm = metric(preds, trues)
        print(
            "accuracy:{}, f1:{}, recall:{}, precision:{}".format(
                acc, f1, recall, precision
            )
        )
        print("confusion matrix:")
        print(cm)

        f = open("result_classification.txt", "a")
        f.write(setting + "  \n")
        f.write(
            "accuracy:{}, f1:{}, recall:{}, precision:{}\n".format(
                acc, f1, recall, precision
            )
        )
        f.write("confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.close()
        return
