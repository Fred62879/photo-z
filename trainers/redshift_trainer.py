import torch
import numpy as np
import torch.nn as nn
import logging as log
import torch.nn.functional as F

from pathlib import Path
from os.path import join
from trainers import BaseTrainer
from trainers.train_utils import *

from lightly.loss import DINOLoss
from lightly.utils.scheduler import cosine_schedule

import warnings
warnings.filterwarnings("ignore")


class RedshiftTrainer(BaseTrainer):
    def __init__(self, model, train_dataset, validation_dataset, optim_cls, optim_params, mode, **kwargs):

        super().__init__(model, train_dataset, validation_dataset, optim_cls, optim_params, mode, **kwargs)
        log.info(f"{self.pipeline}, {next(self.pipeline.parameters()).device}")
        self.warm_up_end = self.kwargs["warm_up_end"]
        self.init_loss()

    def init_loss(self):
        self.dino_loss = DINOLoss(
            output_dim=2048,
            warmup_teacher_temp_epochs=5
        ).to(device)

    ################
    # Train events
    ################

    def train(self):
        self.pre_training()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.begin_epoch()

            for iteration in range(self.num_batches):
                self.iteration = iteration
                self.pre_step()
                data = self.next_batch()
                self.step(data)
                self.post_step()

            self.end_epoch()

        self.post_training()

    def pre_training(self):
        super().pre_training()

        # create dir to save checkpoint and others
        for cur_path, cur_pname, in zip(
                ["model_dir","output_dir"], ["models","outputs"]
        ):
            path = join(self.log_dir, f"{self.cur_pdb_id}_{self.cur_chain_id}", cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        # inform dataset to switch protein chain
        self.train_dataset.set_cur_chain(self.cur_pdb_id, self.cur_chain_id)

    ################
    # Train one epoch
    ################

    def pre_epoch(self):
        self.momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        self.pipeline.train()

    def init_log_dict(self):
        """ Custom log dict.
        """
        super().init_log_dict()

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
        self.optimizer.zero_grad()
        self.add_to_device(data)

        update_momentum(model.student_backbone, model.teacher_backbone, m=self.momentum_val)
        update_momentum(model.student_head, model.teacher_head, m=self.momentum_val)
        views = [view.to(device) for view in data["views"]]
        global_views = views[:2]
        teacher_out = [model.forward_teacher(view) for view in global_views]
        student_out = [model.forward(view) for view in views]
        loss = self.dino_loss(teacher_out, student_out, epoch=self.epoch)
        self.log_dict["total_loss"] += loss.item()

        loss.backward()
        # only cancel gradients of student head.
        model.student_head.cancel_last_layer_gradients(current_epoch=self.epoch)
        self.optimizer.step()

    ############
    # Validation
    ############

    def validate(self):
        if self.mode == "train":
            self.validate_during_train()
        elif self.mode == "validate":
            self.validate_after_train()

    def validate_during_train(self):
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02]
        for thresh in threshs:
            #threshold = self.kwargs["mcubes_threshold"]
            mesh = self.validate_mesh(thresh)
            chamfer_dist = self.calculate_chamfer_dist(mesh)
            log_text = "thresh: {} chamfer distance: {:>.3E}".format(thresh, chamfer_dist)
            log.info(log_text)

    def valid_after_train(self):
        fname = join(self.output_dir,
                     f"output-ep{self.epoch}-it{self.iteration}-thresh{threshold}.ply")


    def validate_mesh(self, threshold):
        resolution = self.kwargs["mesh_resolution"]

        bound_min = torch.tensor(self.train_dataset.get_cur_bbox_mins(),
                                 dtype=torch.float32, device=self.device)
        bound_max = torch.tensor(self.train_dataset.get_cur_bbox_maxs(),
                                 dtype=torch.float32, device=self.device)

        mesh = extract_geometry(
            bound_min, bound_max, self.device, resolution=resolution, threshold=threshold,
            query_func=lambda pts: -self.pipeline.sdf(pts))

        fname = join(self.output_dir,
                     f"output-ep{self.epoch}-it{self.iteration}-thresh{threshold}.ply")
        mesh.export(fname)
        return mesh

    def calculate_chamfer_dist(self, mesh):
        gt_points = self.train_dataset.get_cur_gt_point()
        if self.kwargs["mesh_sample_points"] == -1:
            n = gt_points.shape[0]*gt_points.shape[1]
        else: n = self.kwargs["mesh_sample_points"]
        est_points = sample_mesh(mesh, n, self.kwargs["validate_num_batches"])
        chamfer_dist = calculate_chamfer_dist(
            gt_points, est_points, norm=self.kwargs["chamfer_norm"])
        return chamfer_dist

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

        torch.save(checkpoint, model_fname)
        return checkpoint

    def log_cli(self):
        # Average over iterations
        log_text = 'epoch {}/{}'.format(self.epoch, self.num_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | sdf loss: {:>.3E}'.format(self.log_dict['sdf_loss'] / len(self.train_data_loader))
        log.info(log_text)

    ############
    # Helpers
    ############

    def update_learning_rate(self):
        init_lr = self.kwargs["lr"]
        if self.epoch < self.warm_up_end:
            lr =  (self.epoch / self.warm_up_end)
        else:
            lr = 0.5 * (math.cos((self.epoch - self.warm_up_end) /
                                 (self.num_epochs - self.warm_up_end) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def add_to_device(self, data):
        for field in data:
            data[field] = data[field].to(self.device)