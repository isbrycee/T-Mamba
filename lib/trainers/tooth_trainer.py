import os
import time
import numpy as np
from tqdm import tqdm
import nni
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import math
from lib import utils


class ToothTrainer:
    """
    Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.metric = metric
        self.device = opt["device"]
        if self.opt["use_amp"]:
            self.scaler = GradScaler()

        if not self.opt["optimize_params"]:
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"], utils.datestr() + "_" + opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                utils.make_dirs(self.checkpoint_dir)
                utils.make_dirs(self.tensorboard_dir)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, purge_step=0, max_queue=1, flush_secs=30)

        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_dice = opt["best_dice"]
        self.update_weight_freq = opt["update_weight_freq"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

        self.statistics_dict = self.init_statistics_dict()


    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.reset_statistics_dict()

            self.optimizer.zero_grad()

            self.train_epoch(epoch)

            self.valid_epoch(epoch)

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"])
            else:
                self.lr_scheduler.step()

            print("epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                  .format(epoch, self.end_epoch - 1,
                          self.optimizer.param_groups[0]['lr'],
                          self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                          self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                          self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"],
                          self.best_dice))
            if not self.opt["optimize_params"]:
                utils.pre_write_txt("epoch:[{:03d}/{:03d}]  lr:{:.6f}  train_loss:{:.6f}  train_dsc:{:.6f}  valid_dsc:{:.6f}  best_dsc:{:.6f}"
                                    .format(epoch, self.end_epoch-1,
                                            self.optimizer.param_groups[0]['lr'],
                                            self.statistics_dict["train"]["loss"]/self.statistics_dict["train"]["count"],
                                            self.statistics_dict["train"]["DSC"]["avg"]/self.statistics_dict["train"]["count"],
                                            self.statistics_dict["valid"]["DSC"]["avg"]/self.statistics_dict["valid"]["count"],
                                            self.best_dice), self.log_txt_path)
                self.write_statistcs(mode="epoch", iter=epoch)

            if self.opt["optimize_params"]:
                nni.report_intermediate_result(
                    self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"])

        if self.opt["optimize_params"]:
            nni.report_final_result(self.best_dice)

        time.sleep(60)
        self.writer.close()



    def train_epoch(self, epoch):
        train_start_time = time.time()
        self.model.train()
        
        for batch_idx, (input_tensor, target) in enumerate(self.train_data_loader):

            input_tensor, target = input_tensor.to(self.device), target.to(self.device)

            if self.opt["use_amp"]:
                with autocast():
                    t0 = time.time()
                    output = self.model(input_tensor)
                    t1 = time.time()
                    dice_loss = self.loss_function(output, target)
                    t2 = time.time()

                t3 = time.time()
                self.scaler.scale(dice_loss / self.update_weight_freq).backward()
                t4 = time.time()

                if (batch_idx + 1) % self.update_weight_freq == 0:
                    t5 = time.time()
                    # self.scaler.step(self.optimizer)
                    self.optimizer.step()
                    t6 = time.time()
                    # self.scaler.update()
                    t7 = time.time()
                    self.optimizer.zero_grad()
            else:
                t0 = time.time()
                output = self.model(input_tensor)
                t1 = time.time()
                dice_loss = self.loss_function(output, target)
                t2 = time.time()

                t3 = time.time()
                scaled_dice_loss = dice_loss / self.update_weight_freq
                scaled_dice_loss.backward()
                t4 = time.time()

                if (batch_idx + 1) % self.update_weight_freq == 0:
                    t5 = time.time()
                    self.optimizer.step()
                    t6 = time.time()
                    self.optimizer.zero_grad()

            t8 = time.time()
            # self.calculate_metric_and_update_statistcs(output.cpu().float(), target.cpu().float(), len(target), dice_loss.cpu(), mode="train")
            self.calculate_metric_and_update_statistcs(output.float(), target.float(), len(target), dice_loss, mode="train")

            t9 = time.time()

            if (batch_idx + 1) % self.update_weight_freq == 0 and (not self.opt["optimize_params"]):
                self.write_statistcs(mode="step", iter=epoch*len(self.train_data_loader)+batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                print("epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}"
                      .format(epoch, self.end_epoch-1,
                              batch_idx+1, len(self.train_data_loader),
                              self.optimizer.param_groups[0]['lr'],
                              self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                              self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"]))
                if not self.opt["optimize_params"]:
                    utils.pre_write_txt("epoch:[{:03d}/{:03d}]  step:[{:04d}/{:04d}]  lr:{:.6f}  loss:{:.6f}  dsc:{:.6f}"
                                        .format(epoch, self.end_epoch-1,
                                                batch_idx+1, len(self.train_data_loader),
                                                self.optimizer.param_groups[0]['lr'],
                                                self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                                self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"]),
                                        self.log_txt_path)
        train_end_time = time.time()
        print("train time of one epoch: {:.2f}s".format(train_end_time-train_start_time))


    def valid_epoch(self, epoch):

        self.model.eval()
        print('Start validing...')
        t_start = time.time()
        with torch.no_grad():

            for input_tensor, target in tqdm(self.valid_data_loader):

                input_tensor, target = input_tensor.to(self.device), target.to(self.device)

                output = self.split_forward(input_tensor, self.model)

                # self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target.cpu()), mode="valid")
                self.calculate_metric_and_update_statistcs(output, target, len(target), mode="valid")
            cur_dsc = self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"]

            if (not self.opt["optimize_params"]) and (epoch + 1) % self.save_epoch_freq == 0:
                self.save(epoch, cur_dsc, self.best_dice, type="normal")
            if not self.opt["optimize_params"]:
                self.save(epoch, cur_dsc, self.best_dice, type="latest")
            if cur_dsc > self.best_dice:
                self.best_dice = cur_dsc
                if not self.opt["optimize_params"]:
                    self.save(epoch, cur_dsc, self.best_dice, type="best")
        t_end = time.time() - t_start
        print("valid time:{:.2f}s".format(t_end))


    def write_statistcs(self, mode="step", iter=None):
        if mode == "step":
            self.writer.add_scalar("step_train_loss",
                                   self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                   iter)
            self.writer.add_scalar("step_train_dsc",
                                   self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                                   iter)
        else:
            self.writer.add_scalar("epoch_train_loss",
                                   self.statistics_dict["train"]["loss"] / self.statistics_dict["train"]["count"],
                                   iter)
            self.writer.add_scalar("epoch_train_dsc",
                                   self.statistics_dict["train"]["DSC"]["avg"] / self.statistics_dict["train"]["count"],
                                   iter)
            self.writer.add_scalar("epoch_valid_dsc",
                                   self.statistics_dict["valid"]["DSC"]["avg"] / self.statistics_dict["valid"]["count"],
                                   iter)


    def split_forward(self, image, model):
        ori_shape = image.size()[2:]
        output = torch.zeros((image.size()[0], self.opt["classes"], *ori_shape), device=image.device)
        slice_shape = self.opt["crop_size"]
        stride = self.opt["crop_stride"]

        for shape0_start in tqdm(range(0, ori_shape[0], stride[0])):
            shape0_end = shape0_start + slice_shape[0]
            start0 = shape0_start
            end0 = shape0_end
            if shape0_end >= ori_shape[0]:
                end0 = ori_shape[0]
                start0 = end0 - slice_shape[0]

            for shape1_start in range(0, ori_shape[1], stride[1]):
                shape1_end = shape1_start + slice_shape[1]
                start1 = shape1_start
                end1 = shape1_end
                if shape1_end >= ori_shape[1]:
                    end1 = ori_shape[1]
                    start1 = end1 - slice_shape[1]

                for shape2_start in range(0, ori_shape[2], stride[2]):
                    shape2_end = shape2_start + slice_shape[2]
                    start2 = shape2_start
                    end2 = shape2_end
                    if shape2_end >= ori_shape[2]:
                        end2 = ori_shape[2]
                        start2 = end2 - slice_shape[2]

                    slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                    slice_predict = model(slice_tensor.to(image.device))
                    output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict

                    if shape2_end >= ori_shape[2]:
                        break

                if shape1_end >= ori_shape[1]:
                    break

            if shape0_end >= ori_shape[0]:
                break

        return output
    # def split_forward(self, image, model):
    #     print(next(model.parameters()).device)
    #     model = model.module
    #     ori_shape = image.size()[2:]
    #     output = torch.zeros((image.size()[0], self.opt["classes"], *ori_shape), device=image.device)
    #     slice_shape = self.opt["crop_size"]
    #     stride = self.opt["crop_stride"]
    #     total_slice_num = 1
    #     for i in range(3):
    #         total_slice_num *= math.ceil((ori_shape[i] - slice_shape[i]) / stride[i]) + 1

    #     with tqdm(total=total_slice_num, leave=False) as bar:
    #         for shape0_start in range(0, ori_shape[0], stride[0]):
    #             shape0_end = shape0_start + slice_shape[0]
    #             start0 = shape0_start
    #             end0 = shape0_end
    #             if shape0_end >= ori_shape[0]:
    #                 end0 = ori_shape[0]
    #                 start0 = end0 - slice_shape[0]

    #             for shape1_start in range(0, ori_shape[1], stride[1]):
    #                 shape1_end = shape1_start + slice_shape[1]
    #                 start1 = shape1_start
    #                 end1 = shape1_end
    #                 if shape1_end >= ori_shape[1]:
    #                     end1 = ori_shape[1]
    #                     start1 = end1 - slice_shape[1]

    #                 for shape2_start in range(0, ori_shape[2], stride[2]):
    #                     shape2_end = shape2_start + slice_shape[2]
    #                     start2 = shape2_start
    #                     end2 = shape2_end
    #                     if shape2_end >= ori_shape[2]:
    #                         end2 = ori_shape[2]
    #                         start2 = end2 - slice_shape[2]

    #                     slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
    #                     slice_predict = model(slice_tensor.to(image.device))
    #                     output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict
    #                     bar.update(1)

    #                     if shape2_end >= ori_shape[2]:
    #                         break

    #                 if shape1_end >= ori_shape[1]:
    #                     break

    #             if shape0_end >= ori_shape[0]:
    #                 break

    #     return output

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size, loss=None, mode="train"):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict[mode]["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict[mode]["class_count"][class_name] += cur_batch_size
        if mode == "train":
            self.statistics_dict[mode]["loss"] += loss.item() * cur_batch_size

        mask = mask.to(self.device) # add by hj
        for i, metric_name in enumerate(self.opt["metric_names"]):
            per_class_metric = self.metric[i](output, target)
            per_class_metric = per_class_metric * mask
            self.statistics_dict[mode][metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
            for j, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[mode][metric_name][class_name] += per_class_metric[j].item() * cur_batch_size


    def init_statistics_dict(self):
        statistics_dict = {
            "train": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            },
            "valid": {
                metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
                for metric_name in self.opt["metric_names"]
            }
        }
        for metric_name in self.opt["metric_names"]:
            statistics_dict["train"][metric_name]["avg"] = 0.0
            statistics_dict["valid"][metric_name]["avg"] = 0.0
        statistics_dict["train"]["loss"] = 0.0
        statistics_dict["train"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["valid"]["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["train"]["count"] = 0
        statistics_dict["valid"]["count"] = 0

        return statistics_dict


    def reset_statistics_dict(self):
        for phase in ["train", "valid"]:
            self.statistics_dict[phase]["count"] = 0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[phase]["class_count"][class_name] = 0
            if phase == "train":
                self.statistics_dict[phase]["loss"] = 0.0
            for metric_name in self.opt["metric_names"]:
                self.statistics_dict[phase][metric_name]["avg"] = 0.0
                for _, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[phase][metric_name][class_name] = 0.0


    def save(self, epoch, metric, best_metric, type="normal"):
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.state".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.state'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(state, save_path)
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)


    def load(self):
        if self.opt["resume"] is not None:
            if self.opt["pretrain"] is None:
                raise RuntimeError("Training weights must be specified to continue training")

            resume_state_dict = torch.load(self.opt["resume"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.start_epoch = resume_state_dict["epoch"] + 1
            self.best_dice = resume_state_dict["best_metric"]
            self.optimizer.load_state_dict(resume_state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])

            model_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            if self.opt["pretrain"] is not None:
                model_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
                self.model.load_state_dict(model_state_dict, strict=True)
                print('Succesfully loaded pretrained model: {}.'.format(self.opt["pretrain"]))