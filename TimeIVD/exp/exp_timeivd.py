from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class Exp_TimeIVD(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeIVD, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()  # 预训练 微调
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()

        if self.args.load_checkpoints:  # pretrain: None   finetune: 加载预训练权重
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            transfer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = transfer_weights(self.args.load_checkpoints, model, device=transfer_device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  # todo original
        model_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.args.learning_rate)  # todo verifying
        return model_optim

    def _select_criterion(self):
        if self.args.task_name == "finetune" and self.args.downstream_task == "classification":
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
        # else:
        #     criterion = nn.MSELoss()
        #     print("Using MSELoss")
        else:
            criterion = nn.L1Loss()
            print("Using MAELoss")
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        # model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,T_max=self.args.train_epochs)
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # current learning rate
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            train_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k or "IVD_MOE" in k:  # if "encoder" in k or "enc_embedding" in k:  todo original
                        if "module." in k:
                            k = k.replace("module.", "")  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if "encoder" in k or "enc_embedding" in k:
                        if "module." in k:
                            k = k.replace("module.", "")
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.encoder_state_dict,
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = []
        model_criterion = self._select_criterion()

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            pred_x = self.model(batch_x)
            diff_loss = model_criterion(pred_x, batch_x)  # 计算预训练loss
            diff_loss.backward()

            model_optim.step()
            train_loss.append(diff_loss.item())

        model_scheduler.step()
        train_loss = np.mean(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x,_ = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            # test_loss = self.valid(test_loader, model_criteria)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(  #  Test Loss: {5:.7f}
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    # test_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}\n".format(  #  Test Loss: {5:.7f}
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                        # test_loss,
                    )
                )
            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        self.lr = model_scheduler.get_last_lr()[0]

        return self.model

    def train_1(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # criterion_1 = self._select_criterion_1()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valid(vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x,_ = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()

        return vali_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./outputs/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        historys = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        all_gating_weights = []
        affinity_sums = None
        final_affinity_sums = None
        affinity_sample_count = 0
        case_expert_affinities = None
        case_final_affinities = None

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x, moe_infos = self.model(batch_x)

                if isinstance(moe_infos, (list, tuple)) and len(moe_infos) > 0 and isinstance(moe_infos[0], dict):
                    gating_layers = torch.stack(
                        [layer_info["gating_weights"].detach().cpu() for layer_info in moe_infos],
                        dim=1,
                    ).numpy()
                    expert_layers = torch.stack(
                        [layer_info["expert_affinities"].detach().cpu() for layer_info in moe_infos],
                        dim=1,
                    ).numpy()
                    final_layers = torch.stack(
                        [layer_info["final_affinity"].detach().cpu() for layer_info in moe_infos],
                        dim=1,
                    ).numpy()

                    all_gating_weights.append(gating_layers)

                    if case_expert_affinities is None:
                        case_expert_affinities = expert_layers[:1]
                        case_final_affinities = final_layers[:1]

                    if affinity_sums is None:
                        affinity_sums = expert_layers.sum(axis=0, dtype=np.float64)
                        final_affinity_sums = final_layers.sum(axis=0, dtype=np.float64)
                    else:
                        affinity_sums += expert_layers.sum(axis=0, dtype=np.float64)
                        final_affinity_sums += final_layers.sum(axis=0, dtype=np.float64)
                    affinity_sample_count += expert_layers.shape[0]
                elif torch.is_tensor(moe_infos):
                    all_gating_weights.append(moe_infos.detach().cpu().unsqueeze(1).numpy())  # 0417

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()
                history = batch_x.detach().cpu()

                preds.append(pred)
                trues.append(true)
                historys.append(history)

        if all_gating_weights:
            plot_dir = "./PLOT_wsdm"
            os.makedirs(plot_dir, exist_ok=True)
            plot_prefix = f"{self.args.data}_{self.args.pred_len}"

            final_gating_weights = np.concatenate(all_gating_weights, axis=0)  # [B, L, P, E]
            print(f"所有批次已合并。gating 权重形状: {final_gating_weights.shape}")  # 0417
            np.save(
                os.path.join(plot_dir, f"{plot_prefix}_gating_weights_layers.npy"),
                final_gating_weights,
                allow_pickle=False,
            )

            for layer_idx in range(final_gating_weights.shape[1]):
                np.save(
                    os.path.join(plot_dir, f"{plot_prefix}_layer{layer_idx + 1}_gating_weights_save.npy"),
                    final_gating_weights[:, layer_idx],
                    allow_pickle=False,
                )

            np.save(
                os.path.join(plot_dir, f"{plot_prefix}_gating_weights_save.npy"),
                final_gating_weights[:, -1],
                allow_pickle=False,
            )

            gating_plot_dir = "./PLOT_GatingWeights"
            os.makedirs(gating_plot_dir, exist_ok=True)
            np.save(
                os.path.join(gating_plot_dir, f"{self.args.data}_GatingWeights.npy"),
                final_gating_weights[:, -1],
                allow_pickle=False,
            )

            if affinity_sums is not None and affinity_sample_count > 0:
                mean_expert_affinities = (affinity_sums / affinity_sample_count).astype(np.float32)
                mean_final_affinities = (final_affinity_sums / affinity_sample_count).astype(np.float32)

                np.save(
                    os.path.join(plot_dir, f"{plot_prefix}_MOE_experts_layers_mean.npy"),
                    mean_expert_affinities,
                    allow_pickle=False,
                )
                np.save(
                    os.path.join(plot_dir, f"{plot_prefix}_MOE_final_layers_mean.npy"),
                    mean_final_affinities,
                    allow_pickle=False,
                )

                num_layers = mean_expert_affinities.shape[0]
                num_experts = mean_expert_affinities.shape[2]
                for layer_idx in range(num_layers):
                    np.save(
                        os.path.join(plot_dir, f"{plot_prefix}_layer{layer_idx + 1}_MOE_final_mean.npy"),
                        mean_final_affinities[layer_idx],
                        allow_pickle=False,
                    )
                    for expert_idx in range(num_experts):
                        np.save(
                            os.path.join(plot_dir, f"{plot_prefix}_layer{layer_idx + 1}_MOE_{expert_idx + 1}_mean.npy"),
                            mean_expert_affinities[layer_idx, :, expert_idx],
                            allow_pickle=False,
                        )

                if case_expert_affinities is not None and case_final_affinities is not None:
                    np.save(
                        os.path.join(plot_dir, f"{plot_prefix}_MOE_experts_layers_case.npy"),
                        case_expert_affinities,
                        allow_pickle=False,
                    )
                    np.save(
                        os.path.join(plot_dir, f"{plot_prefix}_MOE_final_layers_case.npy"),
                        case_final_affinities,
                        allow_pickle=False,
                    )

                    for layer_idx in range(case_expert_affinities.shape[1]):
                        np.save(
                            os.path.join(plot_dir, f"{plot_prefix}_layer{layer_idx + 1}_MOE_final.npy"),
                            case_final_affinities[:, layer_idx],
                            allow_pickle=False,
                        )
                        for expert_idx in range(case_expert_affinities.shape[3]):
                            np.save(
                                os.path.join(plot_dir, f"{plot_prefix}_layer{layer_idx + 1}_MOE_{expert_idx + 1}.npy"),
                                case_expert_affinities[:, layer_idx, :, expert_idx],
                                allow_pickle=False,
                            )

                    np.save(
                        os.path.join(plot_dir, f"{plot_prefix}_MOE_final.npy"),
                        case_final_affinities[:, -1],
                        allow_pickle=False,
                    )
                    for expert_idx in range(case_expert_affinities.shape[3]):
                        np.save(
                            os.path.join(plot_dir, f"{plot_prefix}_MOE_{expert_idx + 1}.npy"),
                            case_expert_affinities[:, -1, :, expert_idx],
                            allow_pickle=False,
                        )

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f = open(folder_path + "/score.txt", "a")
        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'history.npy', historys)

        return mse, mae

    def cls_train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_acc = []
            train_f1 = []

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            train_loader = tqdm(train_loader)
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')

                train_loss.append(loss.item())
                train_acc.append(acc)
                train_f1.append(f1)


            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            train_f1 = np.mean(train_f1)

            vali_loss, vali_acc, vali_f1 = self.cls_valid(vali_loader, model_criteria)
            # test_loss, test_acc, test_f1 = self.cls_valid(test_loader, model_criteria)
            self.cls_test(write_log=False)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | ".format(
                    epoch + 1, len(train_loader), end_time - start_time
                ) +
                "Train Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    train_loss, train_acc, train_f1
                ) +
                "Vali Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    vali_loss, vali_acc, vali_f1
                ) #+
                # "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                #     test_loss, test_acc, test_f1
                # )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f}, Acc: {4:.4f}, F1: {5:.4f} | Vali Loss: {6:.7f}, Acc: {7:.4f}, F1: {8:.4f}\n".format(
                        epoch + 1, len(train_loader), end_time - start_time, train_loss, train_acc, train_f1, vali_loss, vali_acc, vali_f1
                    )
                )

            early_stopping(-vali_acc, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def cls_valid(self, vali_loader, model_criteria):
        vali_acc = []
        vali_f1 = []
        vali_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                vali_loss.append(loss.item())
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')
                vali_acc.append(acc)
                vali_f1.append(f1)
        
        vali_loss = np.mean(vali_loss)
        vali_acc = np.mean(vali_acc)
        vali_f1 = np.mean(vali_f1)

        return vali_loss, vali_acc, vali_f1

    def cls_test(self, write_log=True):
        test_data, test_loader = self._get_data(flag="test")
        model_criteria = self._select_criterion()

        preds_all = []
        trues_all = []
        test_loss = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                test_loss.append(loss.item())
                preds_all.extend(preds)
                trues_all.extend(trues)

        test_loss = np.mean(test_loss)
        test_acc = accuracy_score(trues_all, preds_all)
        test_f1 = f1_score(trues_all, preds_all, average='macro')

        print(
            "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                test_loss, test_acc, test_f1
            )
        )
        if write_log:
            f = open(folder_path + "/score.txt", "a")
            f.write(
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}\n".format(test_loss, test_acc, test_f1)
            )
            f.close()

        return test_acc, test_f1
