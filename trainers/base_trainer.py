
import os
import time
import torch
#import wandb
import shutil
import numpy as np
import logging as log

from pathlib import Path
from os.path import join
from datetime import datetime
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler


def log_metric_to_wandb(key, _object, step):
    wandb.log({key: _object}, step=step, commit=False)

def log_images_to_wandb(key, image, step):
    wandb.log({key: wandb.Image(np.moveaxis(image, 0, -1))}, step=step, commit=False)

class BaseTrainer(ABC):
    """ Base class for the trainer. The default overall flow of things:
        init()
        |- set_logger()

        train():
            pre_training()
            (i) for every epoch:
                |- pre_epoch()

                (ii) for every iteration:
                    |- pre_step()
                    |- step()
                    |- post_step()

                post_epoch()
                |- save_model()

                |- validate()

            post_training()
    """

    ################
    # Initialization
    ################

    def __init__(self, pipeline, train_dataset, validation_dataset, optim_cls, optim_params, mode, **kwargs):
        self.kwargs = kwargs

        self.verbose = kwargs["verbose"]
        if kwargs["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(device=self.device)
            log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')
        else: self.device = torch.device("cpu")

        self.mode = mode
        self.using_wandb = False

        # Training params
        self.epoch = 1
        self.iteration = 0
        self.total_iterations = 0
        self.num_epochs = kwargs["num_epochs"]
        self.batch_size = kwargs["batch_size"]
        self.exp_name = kwargs["exp_name"]

        # In-training variables
        self.log_dict = {}
        self.val_data_loader = None
        self.train_dataset_size = None
        self.train_data_loader_iter = None

        self.save_every = kwargs["save_every"]
        self.valid_every = kwargs["valid_every"]
        self.log_tb_every = kwargs["log_tb_every"]
        self.log_cli_every = kwargs["log_cli_every"]
        self.render_tb_every = kwargs["render_tb_every"]

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        if kwargs["dataloader_drop_last"]:
            self.num_batches = len(self.train_dataset) // kwargs["batch_size"]
        else: self.num_batches = int(np.ceil(len(self.train_dataset) / kwargs["batch_size"]))

        self.pipeline = pipeline
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.pipeline.parameters()))
        )

        self.init_dataloader()
        self.init_optimizer(optim_cls, **optim_params)

        self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = join(kwargs["data_path"], "output", self.exp_name, self.log_fname)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Backup configs for debug
        if self.mode[:5] == 'train':
            # self.file_backup()
            dst = join(self.log_dir, "config.yaml")
            shutil.copyfile(kwargs["config"], dst)

    ################
    # Initialization
    ################

    def init_dataloader(self):
        if self.kwargs["shuffle_dataloader"]: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler

        print(len(self.train_dataset))
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler_cls(self.train_dataset), batch_size=self.batch_size,
                drop_last=self.kwargs["dataloader_drop_last"]
            ),
            pin_memory=True,
            num_workers=self.kwargs["dataloader_num_workers"]
        )

        self.iterations_per_epoch = len(self.train_data_loader)

    def init_optimizer(self, optim_cls, **optim_params):
        """ Default initialization for the optimizer.
        """
        params_dict = { name : param for name, param in self.pipeline.named_parameters()}
        params, rest_params = [], []

        for name in params_dict:
            rest_params.append(params_dict[name])

        params.append({"params" : rest_params})
        self.optimizer = optim_cls(params, **optim_params)
        log.info(f"{self.optimizer}")

    #################
    # Data load
    #################

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.train_data_loader_iter = iter(self.train_data_loader)

    def next_batch(self):
        """ Actually iterate the data loader.
        """
        return next(self.train_data_loader_iter)

    def resample_dataset(self):
        """ Override this function if some custom logic is needed.
            Args:
              (torch.utils.data.Dataset): Training dataset.
        """
        if hasattr(self.train_dataset, 'resample'):
            log.info("Reset DataLoader")
            self.train_dataset.resample()
            self.init_dataloader()
        else:
            raise ValueError("resample=True but the dataset doesn't have a resample method")

    ################
    # Training Life-cycle
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

    def is_first_iteration(self):
        return self.total_iterations == 1

    def is_any_iterations_remaining(self):
        return self.total_iterations < self.max_iterations

    def begin_epoch(self):
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()
        self.epoch_start_time = time.time()

    def end_epoch(self):
        current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        self.epoch_start_time = current_time

        self.writer.add_scalar(f'time/elapsed_ms_per_epoch', elapsed_time * 1000, self.epoch)
        if self.using_wandb:
            log_metric_to_wandb(f'time/elapsed_ms_per_epoch', elapsed_time * 1000, self.epoch)

        self.post_epoch()

        if self.valid_every > -1 and self.epoch % self.valid_every == 0 and self.epoch != 0:
            self.validate()

    def save_model(self):
        if self.kwargs["save_as_new"]:
            model_fname = os.path.join(self.log_dir, f'model-ep{self.epoch}-it{self.iteration}.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'model.pth')

        log.info(f'Saving model checkpoint to: {model_fname}')
        if self.kwargs["model_format"] == "full":
            torch.save(self.pipeline, model_fname)
        else:
            torch.save(self.pipeline.state_dict(), model_fname)

        if self.using_wandb:
            name = wandb.util.make_artifact_name_safe(f"{wandb.run.name}-model")
            model_artifact = wandb.Artifact(name, type="model")
            model_artifact.add_file(model_fname)
            wandb.run.log_artifact(model_artifact, aliases=["latest", f"ep{self.epoch}_it{self.iteration}"])

    ################
    # Training Events
    ################

    def pre_training(self):
        """ Override this function to change the logic which runs before the first
              training iteration. This function runs once before training starts.
        """
        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)

        if self.using_wandb:
            wandb_project = self.kwargs["wandb_project"]
            wandb_run_name = self.kwargs.get("wandb_run_name")
            wandb_entity = self.kwargs.get("wandb_entity")
            wandb.init(
                project=wandb_project,
                name=self.exp_name if wandb_run_name is None else wandb_run_name,
                entity=wandb_entity,
                job_type=self.trainer_mode,
                config=self.kwargs,
                sync_tensorboard=True
            )

    def post_training(self):
        """ Override this function to change the logic which runs after the last
              training iteration. This function runs once after training ends.
        """
        self.writer.close()
        if self.using_wandb:
            wandb.finish()

    def pre_epoch(self):
        """ Override this function to change the pre-epoch preprocessing.
            This function runs once before the epoch.
        """

        # The DataLoader is refreshed before every epoch, because by default,
        #   the dataset refreshes (resamples) after every epoch.
        if self.kwargs["resample"] and self.epoch % self.kwargs["resample_every"] == 0 and self.epoch > 1:
            self.resample_dataset()

        self.pipeline.train()

    def post_epoch(self):
        """ Override this function to change the post-epoch post processing.
            By default, this function logs to Tensorboard, renders images to Tensorboard,
              saves the model, and resamples the dataset.

            To keep default behaviour but also augment with other features, do
              `super().post_epoch()` in the derived method.
        """
        self.pipeline.eval()

        total_loss = self.log_dict['total_loss'] / len(self.train_data_loader)
        self.scene_state.optimization.losses['total_loss'].append(total_loss)

        self.log_cli()
        self.log_tb()

        # Render visualizations to tensorboard
        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

       # Save model
        if self.save_every > -1 and self.epoch % self.save_every == 0 and self.epoch != 0:
            self.save_model()

    def pre_step(self):
        """ Override this function to change the pre-step preprocessing (runs per iteration).
        """
        pass

    def post_step(self):
        """ Override this function to change the pre-step preprocessing (runs per iteration).
        """
        pass

    @abstractmethod
    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Params
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        pass

    @abstractmethod
    def validate(self):
        pass

    ################
    # Logging
    ################

    def init_log_dict(self):
        """ Override this function to use custom logs.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0

    def log_model_details(self):
        log.info(f"Position Embed Dim: {self.pipeline.nef.pos_embed_dim}")
        log.info(f"View Embed Dim: {self.pipeline.nef.view_embed_dim}")

    def log_cli(self):
        """ Override this function to change CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))

    def log_tb(self):
        """ Override this function to change loss / other numeric logging to TensorBoard / Wandb.
        """
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f'loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)

    def render_tb(self):
        """ Override this function to change render logging to TensorBoard / Wandb.
        """
        self.pipeline.eval()
        for d in [self.kwargs["num_lods"] - 1]:
            out = self.renderer.shade_images(self.pipeline,
                                             f=self.kwargs["camera_origin"],
                                             t=self.kwargs["camera_lookat"],
                                             fov=self.kwargs["camera_fov"],
                                             lod_idx=d,
                                             camera_clamp=self.kwargs["camera_clamp"])

            # Premultiply the alphas since we're writing to PNG (technically they're already premultiplied)
            if self.kwargs["bg_color"] == 'black' and out.rgb.shape[-1] > 3:
                bg = torch.ones_like(out.rgb[..., :3])
                out.rgb[..., :3] += bg * (1.0 - out.rgb[..., 3:4])

            out = out.image().byte().numpy_dict()

            log_buffers = ['depth', 'hit', 'normal', 'rgb', 'alpha']

            for key in log_buffers:
                if out.get(key) is not None:
                    self.writer.add_image(f'{key}/{d}', out[key].T, self.epoch)
                    if self.using_wandb:
                        log_images_to_wandb(f'{key}/{d}', out[key].T, self.epoch)

    ################
    # Properties
    ################

    @property
    def is_optimization_running(self) -> bool:
        return self.scene_state.optimization.running

    @is_optimization_running.setter
    def is_optimization_running(self, is_running: bool):
        self.scene_state.optimization.running = is_running

    @property
    def epoch(self) -> int:
        """ Epoch counter, starts at 1 and ends at max epochs"""
        return self.cur_epoch

    @epoch.setter
    def epoch(self, epoch: int):
        #self.scene_state.optimization.epoch = epoch
        self.cur_epoch = epoch

    @property
    def iteration(self) -> int:
        """ Iteration counter, for current epoch. Starts at 1 and ends at iterations_per_epoch """
        #return self.scene_state.optimization.iteration
        return self.cur_iteration

    @iteration.setter
    def iteration(self, iteration: int):
        """ Iteration counter, for current epoch """
        #self.scene_state.optimization.iteration = iteration
        self.cur_iteration = iteration

    @property
    def iterations_per_epoch(self) -> int:
        """ How many iterations should run per epoch """
        #return self.scene_state.optimization.iterations_per_epoch
        return self.num_batches

    @iterations_per_epoch.setter
    def iterations_per_epoch(self, iterations: int):
        """ How many iterations should run per epoch """
        #self.scene_state.optimization.iterations_per_epoch = iterations
        log.info("!!!!!!! ALERT !!!!!!!")
        self.num_batches = iterations

    @property
    def total_iterations(self) -> int:
        """ Total iteration steps the trainer took so far, for all epochs.
            Starts at 1 and ends at max_iterations
        """
        return (self.epoch - 1) * self.iterations_per_epoch + self.iteration

    @property
    def max_epochs(self) -> int:
        """ Total number of epochs set for this optimization task.
        The first epoch starts at 1 and the last epoch ends at the returned `max_epochs` value.
        """
        #return self.scene_state.optimization.max_epochs
        return self.num_epochs

    @max_epochs.setter
    def max_epochs(self, num_epochs):
        """ Total number of epochs set for this optimization task.
        The first epoch starts at 1 and the last epoch ends at `num_epochs`.
        """
        #self.scene_state.optimization.max_epochs = num_epochs
        self.num_epochs = num_epochs

    @property
    def max_iterations(self) -> int:
        """ Total number of iterations set for this optimization task. """
        return self.max_epochs * self.iterations_per_epoch

    ############
    # Setters
    ############

    def set_mode(self, mode):
        self.mode = mode

    ############
    # Helpers
    ############

    def file_backup(self):
        dir_lis = self.kwargs['general.recording']
        os.makedirs(join(self.base_exp_dir, 'recording'), exist_ok=True)

        for dir_name in dir_lis:
            cur_dir = join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(join(dir_name, f_name), join(cur_dir, f_name))

        copyfile(self.kwargs['conf_path'], join(self.base_exp_dir, 'recording', 'config.conf'))
