
import sys
import torch
import logging
import torchvision
import numpy as np

from models import DINO
from trainers import RedshiftTrainer
from torch.utils.data import random_split
from dataset import RedshiftDataset, ImageNetDataset

logger_initialized = {}

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

def get_dino_pipeline(**kwargs):
    model = DINO(**kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def get_imagenet_dataset(**kwargs):
    dataset = ImageNetDataset(**kwargs)
    if kwargs["trainer_mode"] == "pre_training":
        return [dataset]

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, range(len(dataset)), kwargs["split_ratio"])
    return [train_dataset, valid_dataset, test_dataset]

def get_redshift_dataset(**kwargs):
    dataset = RedshiftnDataset(**kwargs)
    if kwargs["trainer_mode"] == "pre_training":
        return [dataset]

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, range(len(dataset)), kwargs["split_ratio"])
    return [train_dataset, valid_dataset, test_dataset]

def get_redshift_trainer(model, dataset, optim_cls, optim_params, mode, **kwargs):
    return RedshiftTrainer(model, dataset, optim_cls, optim_params, mode, **kwargs)

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True


    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


'''
def get_optimizer_from_config(args):
    """ Utility function to get the optimizer from the parsed config.
    """
    optim_cls = str2optim[args.optimizer_type]
    if args.optimizer_type == 'adam':
        #optim_params = {'eps': 1e-15}
        optim_params = {'lr': 1e-5, 'eps': 1e-8, 'betas': (args.b1, args.b2),
                        'weight_decay':  args.weight_decay}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}

    return optim_cls, optim_params

def get_dataset_from_config(args):
    """ Utility function to get the dataset from the parsed config.
    """
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"

    transform = None #AddToDevice(device)
    if args.dataset_type == 'astro':
        dataset = AstroDataset(device=device, transform=transform, **vars(args))
        dataset.init()
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return dataset

def get_pipelines_from_config(args, tasks=[]):
    """ Utility function to get the pipelines from the parsed config.
    """
    pipelines = {}
    tasks = set(tasks)
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"

    if args.dataset_type == 'astro':
        nef_train = globals()[args.nef_type](**vars(args))
        pipelines["full"] = AstroPipeline(nef_train)
        log.info(pipelines["full"])

        # pipeline for spectra inferrence
        if "recon_gt_spectra" in tasks or "recon_dummy_spectra" in tasks:
            nef_infer_spectra = globals()[args.nef_type](
                integrate=False, qtz_calculate_loss=False, **vars(args))
            pipelines["spectra_infer"] = AstroPipeline(nef_infer_spectra)

        # pipeline for codebook spectra inferrence
        if "recon_codebook_spectra" in tasks:
            codebook_nef = CodebookNef(integrate=False, **vars(args))
            pipelines["codebook"] = AstroPipeline(codebook_nef)
    else:
        raise ValueError(f"{args.dataset_type} unrecognized dataset_type")

    for _, pipeline in pipelines.items():
        pipeline.to(device)
    return device, pipelines

'''
