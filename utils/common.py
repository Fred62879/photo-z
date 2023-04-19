
import os
import re
import numpy as np

from os.path import join


def bin_data(data, upper_lim, num_bins):
    bins = np.arange(upper_lim, step=upper_lim / (num_bins + 1))
    counts, _ = np.histogram(data, bins=bins)
    return counts

def get_pretrained_model_fname(path=None, model_fname=None, **kwargs):
    log_dir = join(kwargs["log_dir"], kwargs["exp_name"])
    if path is not None:
        pretrained_model_dir = join(log_dir, path)
    else:
        # if log dir not specified, use last directory (exclude newly created one)
        dnames = os.listdir(log_dir)
        assert(len(dnames) > 0)
        dnames.sort()
        pretrained_model_dir = join(log_dir, dnames[-1])

    pretrained_model_dir = join(pretrained_model_dir, "models")

    if model_fname is not None:
        pretrained_model_fname = join(pretrained_model_dir, model_fname)
    else:
        fnames = os.listdir(pretrained_model_dir)
        fnames = sort_alphanumeric(fnames)
        pretrained_model_fname = join(pretrained_model_dir, fnames[-1])
    return pretrained_model_fname

def sort_alphanumeric(arr):
    """ Sort the given iterable in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(arr, key = alphanum_key)
