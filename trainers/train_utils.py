
import torch
import numpy as np
import logging as log
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt

from astropy.stats import mad_std


def cal_photoz(probs, specz_upper_lim, num_specz_bins):
    """ Calculate photometric redshift.
        @Param
          probs: prob of each bin [bsz,nbins]
        @Return
          photozs: [bsz]
    """
    # print(self.kwargs["specz_upper_lim"], self.kwargs["num_specz_bins"])

    bin_width = specz_upper_lim / num_specz_bins
    span = (bin_width/2) + bin_width * np.arange(0, num_specz_bins)
    photozs = np.sum((probs * span), axis=1)
    return photozs

def cal_metrics(photozs, speczs, catastrophic_outlier_thresh):
    """ Calculate metrics to evaluate estimated photo z.
        @Param
          photozs: [bsz]
          speczs: [bsz]
        @Return
          delzs: prediction residual (delta_z)
          madstd: dispersion/MAD deviation
          eta: percent of catastrophic outliers (deltz > th)
    """
    delzs = (photozs - speczs) / (1 + speczs)
    madstd = mad_std(delzs)
    th = 0.05
    eta = np.sum(abs(delzs) > catastrophic_outlier_thresh) / delzs.shape[0]
    return delzs, madstd, eta

def plot_redshift(specz, photoz, fname):

    import mpl_scatter_density # adds projection='scatter_density'
    from matplotlib.colors import LinearSegmentedColormap

    upper_lim = max(specz)
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(specz, photoz, cmap=white_viridis)
    fig.colorbar(density, label='Density of Galaxies (A.U.)')

    plt.xlabel('SpecZ');plt.ylabel('PhotoZ')
    # plt.xlim([0,upper_lim]);plt.ylim([0,upper_lim])
    plt.savefig(fname);plt.close()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    assert 0
    return dist.get_world_size()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        # print(len(self.teacher_temp_schedule))

    def forward(self, student_output, teacher_output, epoch):
        """ Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        # print('*', student_out.shape) # [bsz*ncrops,outdim]
        student_out = student_out.chunk(self.ncrops)
        # for i in student_out:
        #     print(i.shape) # [bsz,outdim]

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # print(teacher_out.shape) # [bsz*2,outdim]
        teacher_out = teacher_out.detach().chunk(2)
        # for i in teacher_out:
        #     print(i.shape) # [bsz,outdim]

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """ Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output)) # * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
