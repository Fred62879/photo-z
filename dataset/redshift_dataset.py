
import os
import torch
import pickle
import numpy as np
import logging as log

from pathlib import Path
from functools import reduce
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from os.path import exists, join
from collections import defaultdict
from torch.utils.data import Dataset

from dataset.data_utils import *


class RedshiftDataset(Dataset):
    def __init__(self, transform=None, **kwargs):
        self.kwargs = kwargs
        self.transform = transform

        self.verbose = kwargs["verbose"]
        if kwargs["use_gpu"]:
            self.device = torch.device('cuda')
        else: self.device = torch.device("cpu")

        self.mode = "pre_training"
        self.load_data_from_cache = kwargs["load_data_from_cache"]
        self.set_log_path()
        self.load_redshift()
        if kwargs["plot_crops"]:
            self.plot_crops()

    def set_log_path(self):
        input_path = join(self.kwargs["data_path"], "input")
        self.source_fits_path = join(input_path, "input_fits")
        self.source_redshift_fname = join(input_path, "redshift", self.kwargs["redshift_fname"])
        self.meta_data_fname = join(input_path, "redshift", "meta_data.txt")
        self.ssl_redshift_data_path = join(input_path, "ssl_redshift")
        Path(self.ssl_redshift_data_path).mkdir(parents=True, exist_ok=True)

    def load_redshift(self):
        """ Read redshift (meta) data.
            Since source table is huge, we need to avoid loading it when possible.

            @Manually set variable
            load_data_from_cache: set to `True` to bypass loading.
              (NOTE this assumes that in the following `crop_path()`, crops and specz
               for all patches are already generated and saved locally. If not, we still
               need to load the dataframe and must set this var to `False`).
        """
        self.data = defaultdict(lambda x: None)

        if self.load_data_from_cache and exists(self.meta_data_fname):
            with open(self.meta_data_fname, "rb") as fp:
                (self.num_crops, self.tracts, self.patches, self.fits_fnames) = pickle.load(fp)
        else:
            df = self.read_source_redshift()
            num_crops = self.crop_patch(df)
            print(num_crops)

            # flatten list
            self.num_crops = reduce(lambda cur, acc: cur + acc, num_crops)
            self.fits_fnames = reduce(lambda cur, acc: cur + acc, self.fits_fnames)

            # save locally
            meta = [self.num_crops, self.tracts, self.patches, self.fits_fnames]
            with open(self.meta_data_fname, "wb") as fp:
                pickle.dump(meta, fp)

        self.total_num_crops = sum(self.num_crops)

    def plot_crops(self):
        for fits_fname in self.fits_fnames:
            crops_fname = join(self.ssl_redshift_data_path, fits_fname[:-5] + "_crops")
            zscale_fname = join(self.ssl_redshift_data_path, fits_fname[:-5] + "_zscale_range.npy")
            crops = np.load(crops_fname + ".npy") # [num_crops,num_bands,sz,sz]
            zscale_range = np.load(zscale_fname)  # [2,num_bands]
            ids = np.random.choice(len(crops), self.kwargs["num_crops_to_plot"], replace=False)
            plot_horizontally(crops[ids], crops_fname + ".png", zscale_ranges=zscale_range)

    def __len__(self):
        if self.mode == "pre_training":
            return self.get_total_num_crops()
        assert 0
        return 0

    def __getitem__(self, idx: list):
        if self.mode == "pre_training":

            if idx[0] == -1:
                idx = self.switch_patch(idx)
                #print(self.cur_crops.shape, self.cur_specz.shape, self.cur_specz_isnull.shape)
            # print(self.cur_crops.shape, idx)

            crops = self.cur_crops[idx]
            if self.transform is not None:
                crops = self.transform(crops)
            return {"crops": crops}

        return {
            "crops": self.crops[index],
            "redshift": self.redshifts[index]
        }

    #############
    # Getters
    #############

    def get_num_crops(self):
        """ returns a list, each value being the #crops of a patch
        """
        return self.num_crops

    def get_total_num_crops(self):
        return self.total_num_crops

    #############
    # Setters
    #############

    def set_mode(self, mode):
        self.mode = mode

    #############
    # Helpers
    #############

    def switch_patch(self, idx):
        num_patches = idx[1]
        patch_ids = idx[2 : 2+num_patches]

        crops, specz, specz_isnull = [], [], []
        for patch_id in patch_ids:
            out_fname_prefix = join(self.ssl_redshift_data_path, self.fits_fnames[patch_id][:-5])
            crops.append(np.load(f"{out_fname_prefix}_crops.npy"))
            specz.append(np.load(f"{out_fname_prefix}_specz.npy"))
            specz_isnull.append(np.load(f"{out_fname_prefix}_specz_isnull.npy"))

        self.cur_crops = np.concatenate(crops, axis=0)
        self.cur_specz = np.concatenate(specz, axis=0)
        self.cur_specz_isnull = np.concatenate(specz_isnull, axis=0)

        return idx[num_patches + 2:]

    def read_source_redshift(self):
        """ Read redshift data from source table (dataframe).
        """
        df = Table.read(self.source_redshift_fname)
        df = df.to_pandas()
        # df.to_parquet('detections_redshifts_pdr3_dud.fits')
        # col_ids = [0,2,4,6,8,-8,-4]

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

        # compile fits filenames
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

        self.tracts = tracts
        self.patches = patches
        self.fits_fnames = fits_fnames
        return df

    def crop_patch(self, df):
        # crop patch at provided center and retrieve specz, save locally
        num_crops = []
        offset = self.kwargs["crop_sz"] // 2

        for i in range(len(self.tracts)):
            cur_tract_num_crops = []
            for j in range(len(self.patches[i])):
                out_fname_prefix = join(self.ssl_redshift_data_path, self.fits_fnames[i][j][:-5])
                cur_crops_fname = f"{out_fname_prefix}_crops.npy"
                cur_specz_fname = f"{out_fname_prefix}_specz.npy"
                cur_specz_isnull_fname = f"{out_fname_prefix}_specz_isnull.npy"
                cur_patch_zscale_range_fname = f"{out_fname_prefix}_zscale_range.npy"

                if exists(cur_crops_fname) and exists(cur_specz_fname) and \
                   exists(cur_specz_isnull_fname) and \
                   exists(cur_patch_zscale_range_fname): continue

                cur_entries = df.loc[ (df["tract"] == self.tracts[i]) &
                                      (df["patch"] == self.patches[i][j]) ]

                # get current spectroscopic redshifts
                if not exists(cur_specz_fname):
                    specz = np.array(list(cur_entries["specz_redshift"])).astype(np.float32)
                    np.save(cur_specz_fname, specz)

                if not exists(cur_specz_isnull_fname):
                    specz_isnull = np.array(list(cur_entries["specz_redshift_isnull"]))
                    np.save(cur_specz_isnull_fname, specz_isnull)

                if exists(cur_crops_fname) and exists(cur_patch_zscale_range_fname):
                    continue

                # crop around each specz center pixel
                ras = list(cur_entries["ra"])
                decs = list(cur_entries["dec"])

                cur_patch_num_crops = 0
                cur_patch, cur_crops = [], []
                r, c, header = None, None, None
                for band in self.kwargs["bands"]:
                    fits_fname = join(self.source_fits_path,
                                      self.fits_fnames[i][j].replace("?", band))
                    hdu = fits.open(fits_fname)[1]
                    if header is None:
                        header = hdu.header
                        wcs = WCS(header)
                        cs, rs = wcs.all_world2pix(ras, decs, 0) # c,r / x,y pixel coord
                        rs = rs.round().astype(int)
                        cs = cs.round().astype(int)
                        cur_patch_num_crops += len(rs)
                    cur_patch.append(hdu.data)

                cur_tract_num_crops.append(cur_patch_num_crops)
                cur_patch = np.array(cur_patch).astype(np.float32)

                # get z-scaling range for current patch
                zscale_ranges = calculate_zscale_ranges(cur_patch)
                np.save(cur_patch_zscale_range_fname, zscale_ranges)

                # generate crops
                # print(self.fits_fnames[i][j], rs, cs)
                for r,c in zip(rs,cs):
                    cur_crops.append(
                        cur_patch[:, r-offset:r+offset, c-offset:c+offset])
                cur_crops = np.array(cur_crops)
                np.save(cur_crops_fname, cur_crops)

            num_crops.append(cur_tract_num_crops)

        return num_crops


# dataset class ends
#####################

def insert_to_string(str1, str2, i):
    """ insert str2 at index i of str1 """
    return str1[:i] + str2 + str1[-i:]
