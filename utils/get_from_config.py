
import torch

from models import DINO, DINOz
from dataset.transforms import *
from torch.utils.data import random_split
from dataset import RedshiftDataset, ImageNetDataset
from trainers import RedshiftTrainer, ImageNetTrainer

from utils.common import get_pretrained_model_fname


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
        pretrained_model_fname = get_pretrained_model_fname(**kwargs)
        model = DINOz(pretrained_model_fname, **kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(model)
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
        dataset = RedshiftDataset(transform=transform, **kwargs)
        return [dataset]

    # transform = transforms.Compose([
    #     SDSSDR12Reddening(deredden=True),
    #     JitterCrop(outdim=params.crop_size),
    #     transforms.ToTensor()]
    # )

    dataset = RedshiftDataset(mode="redshift_train", **kwargs)
    n = len(dataset)

    train_num = int(kwargs["split_ratio"][0] * n)
    valid_num = int(kwargs["split_ratio"][1] * n)
    test_num = n - train_num - valid_num

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [train_num, valid_num, test_num],
        generator=torch.Generator().manual_seed(42)
    )
    return [train_dataset, valid_dataset, test_dataset]

def get_imagenet_trainer(model, dataset, optim_cls, optim_params, mode, **kwargs):
    return ImageNetTrainer(model, dataset, optim_cls, optim_params, mode, **kwargs)

def get_redshift_trainer(model, dataset, optim_cls, optim_params, mode, **kwargs):
    return RedshiftTrainer(model, dataset, optim_cls, optim_params, mode, **kwargs)
