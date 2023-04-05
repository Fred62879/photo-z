import torch
import numpy as np
import torch.nn as nn
import logging as log
import torch.nn.functional as F

from pathlib import Path
from os.path import join
from trainers import BaseTrainer
from trainers.train_utils import *
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

import warnings
warnings.filterwarnings("ignore")


class ImageNetTrainer(BaseTrainer):
    def __init__(self, model, dataset, optim_cls, optim_params, mode, **kwargs):

        super().__init__(model, dataset, optim_cls, optim_params, mode, **kwargs)
        # log.info(f"{self.pipeline}, {next(self.pipeline.parameters()).device}")

        self.mode == mode
        self.init_loss()
        if self.mode == "pre_training":
            self.init_scheduler()

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

        elif self.mode == "redshift_est":
            pass

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

    def init_dataloader(self):
        if self.mode == "pre_training":
            if self.kwargs["shuffle_dataloader"]: sampler_cls = RandomSampler
            else: sampler_cls = SequentialSampler

            self.train_data_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.kwargs["dataloader_drop_last"],
                pin_memory=True,
                num_workers=self.kwargs["dataloader_num_workers"]
            )

            self.iterations_per_epoch = len(self.train_data_loader)

        elif self.mode == "redshift_training":
            if self.kwargs["shuffle_dataloader"]: sampler_cls = RandomSampler
            else: sampler_cls = SequentialSampler

            self.train_data_loader = DataLoader(
                self.dataset[0],
                batch_size=self.batch_size,
                drop_last=self.kwargs["dataloader_drop_last"],
                pin_memory=True,
                num_workers=self.kwargs["dataloader_num_workers"]
            )

            self.iterations_per_epoch = len(self.train_data_loader)

            self.valid_data_loader = DataLoader(
                self.dataset[1],
                batch_size=self.batch_size,
                drop_last=self.kwargs["dataloader_drop_last"],
                pin_memory=True,
                num_workers=self.kwargs["dataloader_num_workers"]
            )

            self.iterations_per_epoch = len(self.train_data_loader)

        # else


    ################
    # Train events
    ################

    def pre_training(self):
        super().pre_training()

        # create dir to save checkpoint and others
        for cur_path, cur_pname, in zip(
                ["model_dir","output_dir"], ["models","outputs"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

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
            self.save_model()

    ############
    # One step
    ############

    def step(self, data):
        if self.mode == "pre_training":
            self.step_pre_training(data)
        elif self.mode == "redshift_est":
            self.step_redshift_est(data)

    # def step_pre_training(self, data):
    #     self.optimizer.zero_grad()
    #     self.add_to_device(data)

    #     self.update_momentum(self.model.student_backbone,
    #                          self.model.teacher_backbone, m=self.momentum_val)
    #     self.update_momentum(self.model.student_head,
    #                          self.model.teacher_head, m=self.momentum_val)

    #     views = [view.to(self.device) for view in data["views"]]
    #     global_views = views[:2]
    #     teacher_out = [self.model.forward_teacher(view) for view in global_views]
    #     student_out = [self.model.forward(view) for view in views]
    #     loss = self.dino_loss(teacher_out, student_out, epoch=self.epoch)
    #     self.log_dict["total_loss"] += loss.item()

    #     loss.backward()
    #     # only cancel gradients of student head.
    #     model.student_head.cancel_last_layer_gradients(current_epoch=self.epoch)
    #     self.optimizer.step()

    def step_pre_training(self, data):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.total_iterations]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[self.total_iterations]

        self.optimizer.zero_grad()

        #for i in data[0]:
        #    print(i.shape)
        #print(data[1:])

        # teacher_output, student_output = self.pipeline(data["images"])
        teacher_output, student_output = self.pipeline(data[0])
        loss = self.dino_loss(student_output, teacher_output, self.epoch)
        loss.backward()
        self.pipeline.prepare_grad_update(self.epoch)
        self.optimizer.step()
        self.pipeline.update_teacher(self.total_iterations, self.momentum_schedule)

    def step_redshift_est(self, data):
        pass

    def step_redshift_est(self, data):
        pass

    ############
    # Validation
    ############

    def validate(self):
        if self.mode == "train":
            self.validate_during_train()
        elif self.mode == "validate":
            self.validate_after_train()

    ############
    # Logging
    ############

    def save_model(self):
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

    def log_cli(self):
        # Average over iterations
        log_text = 'epoch {}/{}'.format(self.epoch, self.num_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log.info(log_text)

    ############
    # Helpers
    ############

    def add_to_device(self, data):
        # images, specz_bin = map(lambda x: x.to(self.device), data[:2])
        for field in data:
            data[field] = data[field].to(self.device)

    ############
    # Setters
    ############

    def set_mode(self, mode):
        self.mode = mode
