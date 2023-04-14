
import torch
import logging as log

from astropy.table import Table
from os.path import exists, join

from models import DINO, DINOz
from dataset.transforms import *
from utils.common import get_pretrained_model_fname
from dataset import RedshiftDataset, ImageNetDataset
from trainers import RedshiftTrainer, ImageNetTrainer


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def get_optimizer(**kwargs):
    """ Utility function to get the optimizer from the parsed config.
    """
    optim_cls = str2optim[kwargs["optimizer_type"]]
    if kwargs["optimizer_type"] == 'adamw':
        optim_params = {'lr': 1e-3} #, 'eps': 1e-8, 'betas': (kwargs["b1"], kwargs["b2"])}
    elif kwargs["optimizer_type"] == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}

    return optim_cls, optim_params

def get_pipeline(**kwargs):
    if kwargs["trainer_mode"] == "pre_training":
        model = DINO(**kwargs)

    elif kwargs["trainer_mode"] == "redshift_train":
        pretrained_model_fname = get_pretrained_model_fname(
            kwargs["pretrained_log_dir"], kwargs["pretrained_model_fname"], **kwargs)
        log.info(f"pretrained model fname: {pretrained_model_fname}")
        model = DINOz(pretrained_model_fname, **kwargs)

    elif kwargs["trainer_mode"] == "test":
        best_model_fname = join(
            kwargs["log_dir"], kwargs["exp_name"], kwargs["best_model_log_dir"],
            "models", kwargs["best_model_fname"])
        model = DINOz(best_model_fname, **kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def get_imagenet_dataset(**kwargs):
    dataset = ImageNetDataset(**kwargs)
    if kwargs["trainer_mode"] == "pre_training":
        return [dataset]

    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, kwargs["split_ratio"], generator=generator)
    return [train_dataset, valid_dataset, test_dataset]

def get_redshift_dataset(**kwargs):
    if kwargs["trainer_mode"] == "pre_training":
        transform = RedshiftDINOTransform(**kwargs)
        train_dataset = RedshiftDataset(
            "pre_training", kwargs["source_table_fname"], transform=transform, **kwargs)
        return train_dataset

    elif kwargs["trainer_mode"] == "redshift_train":
        transform = None
        train_dataset = RedshiftDataset(
            "train_specz", kwargs["train_specz_table_fname"], transform=transform, mode="redshift_train", **kwargs)
        valid_dataset = RedshiftDataset(
            "valid_specz", kwargs["valid_specz_table_fname"], transform=transform, mode="redshift_train", **kwargs)
        return train_dataset, valid_dataset

    elif kwargs["trainer_mode"] == "test":
        transform = None
        test_dataset = RedshiftDataset(
            "test_specz", kwargs["test_specz_table_fname"], transform=transform, mode="test", **kwargs)
        return test_dataset

    else: raise ValueError("Unsupported trainer mode when initializing dataset.")

def get_imagenet_trainer(model, dataset, optim_cls, optim_params, mode, **kwargs):
    return ImageNetTrainer(model, dataset, optim_cls, optim_params, mode, **kwargs)

def get_redshift_trainer(model, dataset, optim_cls, optim_params, mode, **kwargs):
    return RedshiftTrainer(model, dataset, optim_cls, optim_params, mode, **kwargs)

def split_source_table(**kwargs):
    """ Split source data (specz is not None) into 3 tables for train, validation, and test.
    """
    input_path = join(kwargs["data_path"], "input")
    source_table_fname = join(
        input_path, kwargs["redshift_path"], kwargs["source_table_fname"])
    train_table_fname = join(
        input_path, kwargs["redshift_path"], kwargs["train_specz_table_fname"])
    valid_table_fname = join(
        input_path, kwargs["redshift_path"], kwargs["valid_specz_table_fname"])
    test_table_fname = join(
        input_path, kwargs["redshift_path"], kwargs["test_specz_table_fname"])
    if exists(train_table_fname) and exists(valid_table_fname) and exists(test_table_fname):
        return

    df = Table.read(source_table_fname)
    df = df.to_pandas()
    df = df.loc[df["specz_redshift_isnull"] == False]

    n = len(df)
    train_num = int(kwargs["split_ratio"][0] * n)
    valid_num = int(kwargs["split_ratio"][1] * n)
    test_num = n - train_num - valid_num
    # train_num, valid_num, test_num = 4, 2, 2 # for test data only
    log.info(f"redshift train table size: train-{train_num}, valid-{valid_num}, test-{test_num}")

    idx = np.arange(n)
    np.random.shuffle(idx)
    train_table = Table.from_pandas( df.iloc[idx[:train_num]] )
    valid_table = Table.from_pandas( df.iloc[idx[train_num : train_num + valid_num]] )
    test_table = Table.from_pandas( df.iloc[idx[-test_num:]] )

    train_table.write(train_table_fname)
    valid_table.write(valid_table_fname)
    test_table.write(test_table_fname)
