
import os
import torch
import numpy as np

from os.path import exists


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
