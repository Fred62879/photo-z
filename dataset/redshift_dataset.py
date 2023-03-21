
import os
import torch
import numpy as np
import logging as log

from pathlib import Path
from os.path import exists, join
from collections import defaultdict
from torch.utils.data import Dataset

import sys
sys.path.insert(0, './dataset')
from data_utils import *


class RedshiftDataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.verbose = kwargs["verbose"]
        if kwargs["use_gpu"]:
            self.device = torch.device('cuda')
        else: self.device = torch.device("cpu")

        self.mode = "pre_training"
        self.set_log_path()
        self.load_redshift()

    def set_log_path(self):
        root_path = self.kwargs["data_path"]
        input_path = join(root_path, "input")
        self.dataset_path = join(input_path, self.kwargs["dataset_name"])
        self.source_redshift_fname = join(input_path, self.kwargs["redshift_fname"])

    def __len__(self):
        return self.get_num_cutouts()

    def __getitem__(self, idx: list):
        if self.mode == "pre_training":
            ret = {"cutouts": self.cutouts[index]}
        return {
            "cutouts": self.cutouts[index],
            "redshift": self.redshifts[index]
        }

    #############
    # Getters
    #############

    def get_num_cutouts(self):
        return len(self.data["cutouts"])

    #############
    # Setters
    #############

    def set_mode(self, mode):
        self.mode = mode

    #############
    # Helpers
    #############

    def load_redshift(self):
        self.data = defaultdict(lambda x: None)
