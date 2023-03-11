
import os
import csv
import torch
import trimesh
import numpy as np

from os.path import exists
from scipy.spatial import cKDTree
from protein_dataset import ProteinDataset


def read_csv(in_fname):
    chain_ids = {}

    with open(in_fname) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                column_names = row
            else:
                pdb_id = row[0]
                chain_ids[pdb_id] = []
                for chain_id in row[1:]:
                    if chain_id == "": break
                    chain_ids[pdb_id].append(chain_id)

            line_count += 1
    return chain_ids

def save_csv(max_num_chains, data, out_fname):
    try:
        column_names = ["pdb_id"]
        for i in range(max_num_chains):
            column_names.append(f"chain_id_{i}")

        with open(out_fname, 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=column_names)
            writer.writeheader()

            for pdb_id in data:
                row = {"pdb_id":pdb_id}
                for i in range(max_num_chains):
                    if i < len(data[pdb_id]):
                        row[f"chain_id_{i}"] = data[pdb_id][i]
                    else: row[f"chain_id_{i}"] = ""

                writer.writerow(row)

    except IOError:
        print("I/O error")

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

    # normalize point cloud to a unit circle
    shape_scale = np.max([
        np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),
        np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),
        np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])

    shape_center = [
        (np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2,
        (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2,
        (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]

    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    # sample points
    num_points_per_batch = pointcloud.shape[0] // num_batches
    num_gt_points = pointcloud.shape[0] // num_batches * num_batches

    point_idx = np.random.choice(pointcloud.shape[0], num_gt_points, replace = False)
    pointcloud = pointcloud[point_idx,:]

    sigmas = []
    ptree = cKDTree(pointcloud)

    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51) # [num_gt_points//100,3]
        sigmas.append(d[0][:,-1]) # d[0] [num_gt_points//100,51]
    sigmas = np.concatenate(sigmas) # [num_gt_points,]

    sample, sample_near = [], []
    for i in range(query_each):
        scale = 0.25 * np.sqrt(num_gt_points / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * \
            np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(num_batches,num_points_per_batch,3)

        sample_near_tmp = []
        for j in range(num_batches):
            nearest_idx = search_nearest_point(
                torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())

            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)

        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)

    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)
    pointcloud = pointcloud.reshape(num_batches, -1, 3)
    np.savez(out_fname, sample = sample, point = pointcloud, sample_near = sample_near)

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12)
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()
    return dis_idx
