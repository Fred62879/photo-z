
import os
import torch
import numpy as np
import logging as log

from pathlib import Path
from astropy.table import Table
from os.path import exists, join
from collections import defaultdict
from torch.utils.data import Dataset

from dataset.data_utils import *


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
        data_path = self.kwargs["data_path"]
        input_path = join(root_path, "input")
        # self.dataset_path = join(input_path, self.kwargs["dataset_name"])
        self.source_redshift_fname = join(input_path, "redshift", self.kwargs["redshift_fname"])

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

    def insert_to_string(str1, str2, i):
        """ insert str2 at index i of str1 """
        return str1[:i] + str2 + str1[-i:]

    def load_redshift(self):
        self.data = defaultdict(lambda x: None)

        df = Table.read(self.source_redshift_fname)
        df = df.to_pandas()

        # df.to_parquet('detections_redshifts_pdr3_dud.fits')

        col_ids = [0,2,4,6,8,-8,-4]

        # compile fits filenames
        tracts = list(set(df['tract']))
        patches = [
            list(set(
                df.loc[df['tract'] == tract]['patch']
            ))
            for tract in tracts
        ] # [107,..],[...]]

        # patches = [
        #     [insert_to_string(
        #         ("00" + str(patch))[-3:], "%2C", 1)
        #      for patch in cur_patches]
        #     for cur_patches in patches
        # ] # [[1%2C7,..],[...]]

        fits_fnames = []
        for i in range(len(tracts)):
            cur_fits_fnames = []
            for j in range(len(patches[i])):
                patch_name = insert_to_string(
                    ("00" + str(patches[i][j]))[-3:], "%2C", 1)
                cur_fits_fnames.append(
                    "calexp-HSC-?-" + str(tracts[i]) + "-" + patch_name + ".fits"
                )
            fits_fnames.append(cur_fits_fnames)

        # crop patch at provided center, save locally
        for i in range(len(tracts)):
            for j in range(len(patches[i])):
                ras = list(
                    df.loc[ (df["tract"] == tracts[i]) &
                            (df["patch"] == patches[i][j]) ]["ra"])
                decs = list(
                    df.loc[ (df["tract"] == tracts[i]) &
                            (df["patch"] == patches[i][j]) ]["dec"])

                cur_crops = []

                r, c, header = None, None, None
                crops = []
                for band in self.kwargs["bands"]:
                    fits_fname = join(self.source_fits_path,
                                      fits_fnames[i][j].replace("?", band))
                    hdu = fits.open(fits_fname)[1]
                    if header is None:
                        header = hdu.header
                        wcs = WCS(header)
                        c, r = wcs.all_world2pix(ras, decs, 0) # c,r / x,y pixel coord

                    img = hdu.data
                    crops.append(img[
