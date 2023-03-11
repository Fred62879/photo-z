
import os
import csv
import torch
import trimesh
import numpy as np
import logging as log

from pathlib import Path
from os.path import exists, join
from collections import defaultdict
from torch.utils.data import Dataset

import sys
sys.path.insert(0, './dataset')
from data_utils import *


class ProteinDataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.verbose = kwargs["verbose"]
        if kwargs["use_gpu"]:
            self.device = torch.device('cuda')
        else: self.device = torch.device("cpu")

        self.set_log_path()
        self.load_pdb_chain_ids()
        self.load_sample_points_from_pdbs()
        self.set_first_chain()

    def set_log_path(self):
        raw_data_path = self.kwargs["raw_data_path"]
        self.raw_pdb_path = join(raw_data_path, self.kwargs["raw_pdb_dir"])
        self.raw_surface_path = join(raw_data_path, self.kwargs["raw_surface_dir"])

        root_path = self.kwargs["data_path"]
        input_path = join(root_path, "input")
        self.dataset_path = join(input_path, self.kwargs["dataset_name"])
        self.pdb_chain_ids_fname = join(self.dataset_path, self.kwargs["pdb_chain_id_name"] + ".txt")

    def load_pdb_chain_ids(self):
        if exists(self.pdb_chain_ids_fname):
            #with open(self.pdb_chain_ids_fname, "rb") as fp:
            #    self.chain_ids = pickle.load(fp)
            self.chain_ids = read_csv(self.pdb_chain_ids_fname)
        else:
            #with open (self.pdb_chain_ids_fname, "wb") as fp:
            #    pickle.dump(self.chain_ids, fp)
            max_num_chains = self.gather_raw_data_pdb_chain_ids()
            save_csv(max_num_chains, self.chain_ids, self.pdb_chain_ids_fname)

        self.pdb_ids = list(self.chain_ids.keys())

    def load_sample_points_from_pdbs(self):
        self.points, self.samples, self.gt_points = {}, {}, {}
        self.bbox_mins, self.bbox_maxs = {}, {}
        for pdb_id in self.pdb_ids:
            for chain_id in self.chain_ids[pdb_id]:
                chain_dir = join(self.dataset_path, f"{pdb_id}_{chain_id}")
                Path(chain_dir).mkdir(parents=True, exist_ok=True)
                self.load_sample_points_from_one_chain(pdb_id, chain_id)

    def set_first_chain(self):
        """ Expose the first protein chain. """
        pdb_id = self.pdb_ids[0]
        chain_id = self.chain_ids[pdb_id][0]
        self.cur_chain = f"{pdb_id}_{chain_id}"

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

    def get_cur_gt_point(self):
        return self.gt_points[self.cur_chain]

    def get_cur_num_sample_points(self):
        return len(self.samples[self.cur_chain]) - 1

    def get_chain_ids(self):
        return self.chain_ids

    def get_cur_bbox_mins(self):
        return self.bbox_mins[self.cur_chain]

    def get_cur_bbox_maxs(self):
        return self.bbox_maxs[self.cur_chain]

    #############
    # Setters
    #############

    def set_cur_chain(self, pdb_id, chain_id):
        self.cur_chain = f"{pdb_id}_{chain_id}"

    #############
    # Helpers
    #############

    def gather_raw_data_pdb_chain_ids(self):
        """ Collect pdb ids from pdb file directory.
            Pdb files are separated into single chain file.
        """
        fnames = os.listdir(self.raw_pdb_path)

        # get chain id for each pdb
        self.chain_ids = defaultdict(lambda: [])
        for fname in fnames:
            splits = fname.split("_")
            pdb_id = splits[0]
            chain_id = splits[1].split(".")[0]
            self.chain_ids[pdb_id].append(chain_id)

        # sort chain ids for each pdb
        for pdb_id in self.chain_ids:
            tmp = self.chain_ids[pdb_id]
            tmp.sort()
            self.chain_ids[pdb_id] = tmp

        # **** tmp for testing **** !!! remove afterwards !!!
        pdb_ids = list(self.chain_ids.keys())
        pdb_ids.sort()
        pdb_ids = pdb_ids[:1]
        self.chain_ids = { pdb_id: self.chain_ids[pdb_id] for pdb_id in pdb_ids}
        # **** ****

        max_num_chains = 0
        for pdb_id in self.chain_ids:
            max_num_chains = max(max_num_chains, len(self.chain_ids[pdb_id]))

        # gather all pdbs
        # self.pdb_ids = np.array(list(set([fname[:4] for fname in fnames])))
        # self.pdb_ids.sort()
        return max_num_chains

    def load_sample_points_from_one_chain(self, pdb_id, chain_id):
        key = f"{pdb_id}_{chain_id}"
        out_fname = join(self.dataset_path, key, f"{key}.npz")

        if exists(out_fname):
            if self.verbose: log.info(f"loading {key}")
        else:
            if self.verbose: log.info(f"processing {key}")
            in_fname = join(self.raw_surface_path, f"{key}")
            process_data(in_fname, out_fname, num_batches=self.kwargs["validate_num_batches"])

        processed_data = np.load(out_fname)

        # sampled point and nearest point on surface for training
        point = np.asarray(processed_data['sample_near']).reshape(-1,3).astype("float32") #997500
        sample = np.asarray(processed_data['sample']).reshape(-1,3).astype("float32") #997500

        # gt point for validation purpose
        gt_point = np.asarray(processed_data['point']).astype("float32") #5700

        object_bbox_min = np.array([np.min(point[:,0]), np.min(point[:,1]), np.min(point[:,2])]) -0.05
        object_bbox_max = np.array([np.max(point[:,0]), np.max(point[:,1]), np.max(point[:,2])]) +0.05
        if self.verbose: log.info(f"data bounding box: {object_bbox_min} {object_bbox_max}")

        self.points[key] = point
        self.samples[key] = sample
        self.gt_points[key] = gt_point
        self.bbox_mins[key] = object_bbox_min
        self.bbox_maxs[key] = object_bbox_max
