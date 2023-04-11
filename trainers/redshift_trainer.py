
import torch
import numpy as np
import torch.nn as nn
import logging as log
import torch.nn.functional as F

from pathlib import Path
from os.path import join
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from trainers import BaseTrainer
from trainers.train_utils import *
from dataset.samplers import PatchWiseSampler
from utils.common import get_pretrained_model_fname


class RedshiftTrainer(BaseTrainer):
    def __init__(self, model, dataset, optim_cls, optim_params, mode, **kwargs):
        super().__init__(model, dataset, optim_cls, optim_params, mode, **kwargs)
        # log.info(f"{self.pipeline}, {next(self.pipeline.parameters()).device}")

        self.mode == mode
        self.init_dataset(dataset)
        self.init_dataloader()
        self.init_loss()

    def set_log_path(self):
        super().set_log_path()
        if self.mode == "redshift_train":
            self.best_model_fname = join(self.model_dir, self.kwargs["best_model_fname"])

    def init_dataset(self, dataset):
        if self.mode == "pre_training":
            self.init_scheduler()
            self.train_dataset = dataset
            self.batch_size = self.kwargs["pretrain_batch_size"]
            log.info(f"pretrain dataset length: {len(self.train_dataset)}")

        elif self.mode == "redshift_train":
            self.train_dataset, self.valid_dataset = dataset
            self.batch_size = self.kwargs["pretrain_batch_size"]
            log.info(f"train dataset length: {len(self.train_dataset)}")
            log.info(f"valid dataset length: {len(self.valid_dataset)}")

        elif self.mode == "test":
            self.test_dataset = dataset
            log.info(f"test dataset length: {len(self.test_dataset)}")

        else: raise ValueError("Unsupported trainer mode.")

    def init_dataloader(self):
        if self.mode == "pre_training":
            self.train_data_loader = self._init_dataloader(self.train_dataset)
            self.iterations_per_epoch = int(np.ceil(
                len(self.train_data_loader) / self.batch_size))

        elif self.mode == "redshift_train":
            self.train_data_loader = self._init_dataloader(self.train_dataset)
            self.valid_data_loader = self._init_dataloader(self.valid_dataset)
            self.iterations_per_epoch = int(np.ceil(
                len(self.train_data_loader) / self.batch_size))

        elif self.mode == "test":
            self.valid_data_loader = self._init_dataloader(self.test_dataset)

        else: raise ValueError("Unsupported trainer mode.")

    def init_loss(self):
        if self.mode == "pre_training":
            self.dino_loss = DINOLoss(
                self.kwargs["out_dim"],
                self.kwargs["local_crops_number"] + 2,  # total number of crops = 2 global crops + local_crops_number
                self.kwargs["warmup_teacher_temp"],
                self.kwargs["teacher_temp"],
                self.kwargs["warmup_teacher_temp_epochs"],
                self.kwargs["num_epochs"],
            ).to(self.device)

        elif self.mode == "redshift_train":
            self.redshift_loss = nn.CrossEntropyLoss().to(self.device)

    def init_scheduler(self):
        self.lr_schedule = cosine_scheduler(
            self.kwargs["lr"] * (self.kwargs["batch_size_per_gpu"] * get_world_size()) / 256., # linear scaling
            self.kwargs["min_lr"],
            self.num_epochs, len(self.train_data_loader),
            warmup_epochs=self.kwargs["warmup_epochs"],
        )
        self.wd_schedule = cosine_scheduler(
            self.kwargs["weight_decay"],
            self.kwargs["weight_decay_end"],
            self.num_epochs, len(self.train_data_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = cosine_scheduler(
            self.kwargs["momentum_teacher"], 1, self.num_epochs, len(self.train_data_loader))

    ################
    # Train events
    ################

    def pre_training(self):
        super().pre_training()
        if self.mode == "redshift_train":
            self.best_valid_acc = -1

    ################
    # Train one epoch
    ################

    def pre_epoch(self):
        self.pipeline.train()

    def post_epoch(self):
        self.pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)

        if self.log_tb_every > -1 and self.epoch % self.log_tb_every == 0:
            self.log_tb()

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self._save_model()

    ############
    # One step
    ############

    def step(self, data):
        if self.mode == "pre_training":
            self.step_pre_training(data)
        elif self.mode == "redshift_train":
            self.step_redshift_train(data)
        else: raise ValueError("Unsupported trainer mode for stepping.")

    def step_pre_training(self, data):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.total_iterations]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[self.total_iterations]

        self.optimizer.zero_grad()
        self._add_to_device(data["crops"])

        teacher_output, student_output = self.pipeline(data["crops"])
        loss = self.dino_loss(student_output, teacher_output, self.epoch)
        self.log_dict["total_loss"] += loss.item()
        loss.backward()

        self.pipeline.prepare_grad_update(self.epoch)
        self.optimizer.step()
        self.pipeline.update_teacher(self.total_iterations, self.momentum_schedule)

    def step_redshift_train(self, data):
        self.optimizer.zero_grad()
        self._add_to_device(data)

        logits = self.pipeline(data["crops"])
        # print(logits.shape, data["specz_bin"])
        loss = self.redshift_loss(logits, data["specz_bin"])
        self.log_dict["total_loss"] += loss.item()

        loss.backward()
        self.optimizer.step()

    ############
    # Validation
    ############

    @torch.no_grad()
    def validate(self, data_loader=None, mode="valid"):
        """ Estimating the performance of model on the given dataset.
            @Param
              mode: valid (during training, we validate every certain epochs
                           with current model on valid dataset)
                    test  (after training is done, we valida with best model
                           on test dataset)
        """
        if data_loader is None:
            data_loader = self.valid_data_loader
        self.pipeline.eval()

        num_correct, num_samples = 0, 0
        for data in data_loader:
            self._add_to_device(data)

            bsz = len(data["crops"])
            output = self.pipeline(data["crops"]) # [bsz,num_bins]
            _, preds = output.max(dim=-1)
            num_correct += preds.eq(data["specz_bin"]).sum()
            num_samples += bsz

        avg_acc = num_correct / num_samples

        if mode == "valid":
            if avg_acc > self.best_valid_acc:
                self.best_valid_acc = avg_acc
                self._save_model(self.best_model_fname)

            self.pipeline.train() # switch model back to training mode

        log_text = ""
        if self.mode == "pre_training" or self.mode == "redshift_train":
            log_text = "epoch {}/{} | ".format(self.epoch, self.num_epochs)
        log_text += "validation accuracy: {:>.3E}".format(avg_acc)
        log.info(log_text)

    ############
    # Logging
    ############

    def log_cli(self):
        # Average over iterations
        log_text = 'epoch {}/{}'.format(self.epoch, self.num_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log.info(log_text)

    ############
    # Setters
    ############

    def set_mode(self, mode):
        self.mode = mode

    ############
    # Helpers
    ############

    def _init_dataloader(self, dataset):
        sampler = PatchWiseSampler(
            dataset,
            self.kwargs["pretrain_batch_size"],
            self.kwargs["num_patches_per_group"])
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            pin_memory=True,
            num_workers=self.kwargs["dataloader_num_workers"]
        )
        return data_loader

    def _add_to_device(self, data):
        # images, specz_bin = map(lambda x: x.to(self.device), data[:2])

        if type(data) == torch.Tensor:
            data = data.to(self.device)
        elif type(data) == list:
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
        elif type(data) == dict:
            for field in data:
                data[field] = data[field].to(self.device)
        else:
            assert 0

    def _save_model(self, model_fname=None):
        if model_fname is None:
            fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
            model_fname = join(self.model_dir, fname)

        if self.verbose: log.info(f"saving model checkpoint to: {model_fname}")

        checkpoint = {
            "epoch_trained": self.epoch,
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        if self.mode == "pre_training":
            checkpoint["dino_loss"] = self.dino_loss.state_dict()

        torch.save(checkpoint, model_fname)
        return checkpoint

    def _load_model(self, model_fname):
        checkpoint = torch.load(model_fname)
        self.pipeline.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
