
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from os.path import exists
from astropy.visualization import ZScaleInterval

def process_data(in_fname, out_fname, query_each=25, num_batches=60):
    """ Sample points from point cloud.
        @Param
          in_fname: input pointcloud filename
          out_fname: output filename
          query_each: number of points to sample around each gt point
          num_batches: divide sampled point into batches for nearest point search
        @Return
          sample: sampled points [QUERY_EACH,POINT_NUM_GT,3]
          point: gt points [POINT_NUM_GT,3]
          sample_near: nearest gt point for each sampled point [QUERY_EACH,POINT_NUM_GT,3]
    """
    if exists(in_fname + ".ply"):
        pointcloud = trimesh.load(in_fname + ".ply").vertices
        pointcloud = np.asarray(pointcloud)
    elif exists(in_fname + ".xyz"):
        pointcloud = np.load(in_fname + ".xyz")
    else:
        log.info("only support .xyz or .ply data. Please make adjust your data.")
        exit()

def calculate_zscale_ranges(img):
    """ Calculate zscale ranges based on given pixels for each bands separately.
        @Param
          pixels: [nbands,sz,sz]
        @Return
          zscale: [2,nbands] (vmin, vmax)
    """
    num_bands = img.shape[0]
    zmins, zmaxs = [], []
    for i in range(num_bands):
        zmin, zmax = ZScaleInterval(contrast=.25).get_limits(img[i])
        zmins.append(zmin);zmaxs.append(zmax)
    return np.array([zmins, zmaxs])

def plot_horizontally(imgs, png_fname, plot_option="plot_img", zscale_ranges=None, save_close=True):
    """ Plot multiband image horizontally.
        Currently only supports plot one row.
        @Param
          img: multiband image [nimgs,nbands,sz,sz]
          zscale_ranges: min and max value for zscaling [2,nbands]
    """
    if plot_option == "plot_img":
        if zscale_ranges is None:
            vmins, vmaxs = [], []
            cal_z_range = True
        else:
            (vmins, vmaxs) = zscale_ranges
            cal_z_range = False
    else: vmins, vmaxs, cal_z_range = None, None, False

    num_imgs, num_bands, _, _ = imgs.shape
    fig = plt.figure(figsize=(3*num_bands + 1,3*num_imgs + 1))

    for i, img in enumerate(imgs):
        plot_one_row(fig, num_imgs, num_bands, i*num_bands, img, num_bands,
                     plot_option, vmins, vmaxs, cal_z_range=cal_z_range)

    fig.tight_layout()
    if save_close:
        plt.savefig(png_fname)
        plt.close()

def plot_one_row(fig, r, c, lo, img, num_bands, plot_option, vmins, vmaxs, cal_z_range=False):
    """ Plot current img (multiband) in one row based on plot_option
        @Param
           fig: global figure
           r/c: size of fig
           lo: starting subfigure position
    """
    if plot_option == "plot_img" and cal_z_range:
        for i in range(num_bands):
            vmin, vmax = ZScaleInterval(contrast=.25).get_limits(img[i])
            vmins.append(vmin);vmaxs.append(vmax)

    for i in range(num_bands):
        ax = fig.add_subplot(r, c, lo+i+1)
        if plot_option == "plot_img":
            plot_zscale(ax, img[i], vmins[i], vmaxs[i])
        elif plot_option == "plot_distrib":
            plot_img_distrib_one_band(ax, img[i])

    if cal_z_range:
        return vmins,vmaxs

def plot_zscale(ax, data, vmin, vmax):
    ax.axis('off')
    ax.imshow(data, cmap='gray', interpolation='none', vmin=vmin,
              vmax=vmax, alpha=1.0, aspect='equal',origin='lower')
