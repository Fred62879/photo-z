
import os
import torch
import numpy as np
import logging as log

from pathlib import Path
from os.path import exists, join
from collections import defaultdict
from torch.utils.data import Dataset
from lightly.data import DINOCollateFunction, LightlyDataset

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

        self.set_log_path()

    def set_log_path(self):
        raw_data_path = self.kwargs["raw_data_path"]
        self.raw_pdb_path = join(raw_data_path, self.kwargs["raw_pdb_dir"])
        self.raw_surface_path = join(raw_data_path, self.kwargs["raw_surface_dir"])

        root_path = self.kwargs["data_path"]
        input_path = join(root_path, "input")
        self.dataset_path = join(input_path, self.kwargs["dataset_name"])
        self.pdb_chain_ids_fname = join(self.dataset_path, self.kwargs["pdb_chain_id_name"] + ".txt")

    def __len__(self):
        return self.get_cur_num_sample_points()

    def __getitem__(self, idx: list):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(
            self.get_cur_num_sample_points() //10, len(idx), replace = False)
        index = index_fine * 10 + index_coarse

        return {
            "points": self.points[self.cur_chain][index],
            "samples": self.samples[self.cur_chain][index],
            #"gt_points": self.gt_points[self.cur_chain],
        }

    #############
    # Getters
    #############


    #############
    # Setters
    #############

    #############
    # Helpers
    #############
