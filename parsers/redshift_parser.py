
import torch
import logging
import argparse

from torchvision import models as torchvision_models
from parsers.parsers import parse_input_config, default_log_setup


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def parse_args():
    """ Parse all command arguments and generate all needed ones. """
    parser = define_cmd_line_args()
    config, args_str = parse_input_config(parser)
    args = argparse.Namespace(**config)
    default_log_setup(args.log_level)
    return args, args_str

def define_cmd_line_args():
    """ Define all command line arguments
    """
    parser = argparse.ArgumentParser('DINO', add_help=False)

    torch.hub.set_dir('/scratch/projects/vision/code/dino/cache')

    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group("global")

    global_group.add_argument("--config", type=str, help="Path to config file to replace defaults.")

    global_group.add_argument("--use_gpu", action="store_true", default=True)
    global_group.add_argument("--verbose", action="store_true")
    global_group.add_argument("--print-shape", action="store_true")
    global_group.add_argument("--exp-name", type=str, default="image_net",
                              help="Experiment name.")

    ###################
    # General global network things
    ###################
    net_group = parser.add_argument_group("net")

    net_group.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit',
                                 'deit_tiny', 'deit_small'] + \
                        torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with \
                        ViTs, we recommend using vit_tiny or vit_small.""")
    net_group.add_argument('--in-chans', default=3, type=int)
    net_group.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 \
                        (for 16x16 patches). Using smaller values leads to better \
                        performance but requires more memory. Applies only for ViTs \
                        (vit_tiny, vit_small and vit_base). If <16, we recommend disabling \
                        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    net_group.add_argument('--out_dim', default=65536, type=int,
                        help="""Dimensionality of the DINO head output. For complex and \
                        large datasets large values (like 65k) work well.""")
    net_group.add_argument('--norm_last_layer', default=True, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of \
                        the DINO head. Not normalizing leads to better performance but \
                        can make the training unstable. In our experiments, we typically \
                        set this paramater to False with vit_small and True with vit_base.""")
    net_group.add_argument('--use_bn_in_head', default=False, type=bool_flag,
                        help="Whether to use batch normalizations in \
                        projection head (Default: False)")

    net_group.add_argument('--momentum_teacher', default=0.996, type=float,
                        help="""Base EMA parameter for teacher update. The value is \
                        increased to 1 during training with cosine schedule. We \
                        recommend setting a higher value with small batches: for \
                        example use 0.9995 with batch size of 256.""")
    net_group.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works \
                        well in most cases. Try decreasing it if the training loss \
                        does not decrease.""")
    net_group.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature. \
                        For most experiments, anything above 0.07 is unstable. We recommend\
                        starting with the default value of 0.04 and increase this slightly \
                        if needed.""")
    net_group.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                           help='Number of warmup epochs for the teacher temperature \
                           (Default: 30).')

    ###################
    # Arguments for training
    ###################

    train_group = parser.add_argument_group("trainer")

    train_group.add_argument("--trainer-mode", type=str, choices=[
        "pre_training", "redshift_train", "redshift_test"])
    train_group.add_argument("--save-every", type=int, default=100,
                             help="Save the model at every N epoch.")
    train_group.add_argument("--render-tb-every", type=int, default=-1,
                             help="Render every N iterations")
    train_group.add_argument("--log-tb-every", type=int, default=-1,
                             help="Log to tensorboard at every N epoch.")
    train_group.add_argument("--log-cli-every", type=int, default=-1,
                             help="Log to command line at every N epoch.")
    train_group.add_argument("--log-gpu-every", type=int, default=-1,
                             help="Log to cli gpu usage at every N epoch.")
    train_group.add_argument("--save-local-every", type=int, default=-1,
                             help="Save data to local every N epoch.")
    train_group.add_argument("--gpu-data", nargs="+", type=str,
                             help="data fields that can be added to gpu.")

    train_group.add_argument("--resume-train", action="store_true")
    train_group.add_argument("--resume-log-dir", type=str)
    train_group.add_argument("--resume-model-name", type=str)
    train_group.add_argument("--pretrained-log-dir", type=str)
    train_group.add_argument("--pretrained-model-fname", type=str)

    train_group.add_argument("--log-dir", type=str, default="_results/logs/runs/",
                             help="Log file directory for checkpoints.")

    train_group.add_argument('--num_epochs', default=100, type=int,
                             help='Number of epochs of training.')
    train_group.add_argument("--warmup_epochs", default=10, type=int,
                             help="Number of epochs for the linear learning-rate warm up.")
    train_group.add_argument('--pretrain-batch-size', default=64, type=int)
    train_group.add_argument('--batch_size_per_gpu', default=64, type=int,
                             help='Per-GPU batch-size : number of distinct images \
                             loaded on one GPU.')
    train_group.add_argument('--use_fp16', type=bool_flag, default=True,
                             help="""Whether or not to use half precision for training. \
                             Improves training time and memory requirements, but can provoke \
                             instability and slight decay of performance. We recommend \
                             disabling mixed precision if the loss is unstable, if reducing \
                             the patch size or if training with bigger ViTs.""")
    train_group.add_argument('--weight_decay', type=float, default=0.04,
                             help="""Initial value of the weight decay. With ViT, \
                             a smaller value at the beginning of training works well.""")
    train_group.add_argument('--weight_decay_end', type=float, default=0.4,
                             help="""Final value of the weight decay. We use a cosine \
                             schedule for WD and using a larger decay by the end of \
                             training improves performance for ViTs.""")
    train_group.add_argument('--clip_grad', type=float, default=3.0,
                             help="""Maximal parameter gradient norm if using gradient \
                             clipping. Clipping with norm .3 ~ 1.0 can help optimization \
                             for larger ViT architectures. 0 for disabling.""")
    train_group.add_argument('--freeze_last_layer', default=1, type=int,
                             help="""Number of epochs during which we keep the output \
                             layer fixed. Typically doing so during the first epoch helps \
                             training. Try increasing this value if the loss does not \
                             decrease.""")
    train_group.add_argument('--drop_path_rate', type=float, default=0.1,
                             help="stochastic depth rate")

    ###################
    # Arguments for optimizer
    ###################

    optim_group = parser.add_argument_group("optimizer")

    optim_group.add_argument("--lr", default=0.0005, type=float,
                             help="""Learning rate at the end of linear warmup \
                             (highest LR used during training). The learning rate \
                             is linearly scaled with the batch size, and specified \
                             here for a reference batch size of 256.""")
    optim_group.add_argument('--min_lr', type=float, default=1e-6,
                             help="""Target LR at the end of optimization. \
                             We use a cosine LR schedule with linear warmup.""")
    optim_group.add_argument('--optimizer_type', default='adamw', type=str,
                             choices=['adamw', 'sgd', 'lars'],
                             help="""Type of optimizer. We recommend using adamw with ViTs.""")

    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group("dataset")

    data_group.add_argument("--load-data-from-cache", action="store_true")
    data_group.add_argument("--plot-crops", action="store_true")
    data_group.add_argument("--num-crops-to-plot", type=int, default=1)
    data_group.add_argument("--shuffle-dataloader", action="store_true")
    data_group.add_argument("--dataloader-drop-last", action="store_true")
    data_group.add_argument("--dataloader-num-workers", type=int, default=0)
    data_group.add_argument("--num-patches-per-group", type=int, default=1)

    data_group.add_argument("--dino-global-crop-dim", type=int, default=1)
    data_group.add_argument("--dino-local-crop-dim", type=int, default=1)
    data_group.add_argument("--dino-num-local-crops", type=int, default=1)
    data_group.add_argument("--dino-jitter-lim", type=int, default=1)
    data_group.add_argument("--dino-rotate-mode", type=str, default="wrap")

    data_group.add_argument("--split-ratio", type=int, nargs="+")

    data_group.add_argument("--data-path", type=str, help="Path to the dataset")
    data_group.add_argument("--redshift-fname", type=str, help="Filename of source redshift.")
    data_group.add_argument("--dataset-num-workers", type=int, default=-1,
                            help="Number of workers for dataset preprocessing, if it \
                            supports multiprocessing. -1 indicates no multiprocessing.")
    data_group.add_argument("--crop-sz", type=str, help="Size of crop.")
    data_group.add_argument("--bands", type=str, nargs="+", help="Band of images.")
    data_group.add_argument("--sensor-collection-name", type=str, help="Choice of band col.")

    data_group.add_argument('--saveckp_freq', default=20, type=int,
                            help='Save checkpoint every x epochs.')
    data_group.add_argument('--seed', default=0, type=int, help='Random seed.')
    data_group.add_argument("--dist_url", default="env://", type=str,
                            help="""url used to set up distributed training; \
                            see https://pytorch.org/docs/stable/distributed.html""")
    data_group.add_argument("--local_rank", default=0, type=int,
                            help="Please ignore and do not set this argument.")
    data_group.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                            help="""Scale range of the cropped image before resizing,
                            relatively to the origin image. Used for large global view cropping.
                            When disabling multi-crop (--local_crops_number 0), we recommand \
                            using a wider range of scale ("--global_crops_scale 0.14 1." \
                            for example)""")
    data_group.add_argument('--local_crops_number', type=int, default=8,
                            help="""Number of small local views to generate. Set this \
                            parameter to 0 to disable multi-crop training. When disabling \
                            multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    data_group.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                            help="""Scale range of the cropped image before resizing, \
                            relatively to the origin image. Used for small local view \
                            cropping of multi-crop.""")

    ###################
    # Arguments for validation
    ###################
    valid_group = parser.add_argument_group("validation")

    valid_group.add_argument("--valid-only", action="store_true",
                             help="Run validation only (and do not run training).")
    valid_group.add_argument("--valid-every", type=int, default=-1,
                             help="Frequency of running validation.")
    valid_group.add_argument("--valid-split", type=str, default="val",
                             help="Split to use for validation.")
    valid_group.add_argument("--validate_num_batches", type=int, default=10)


    # add log level flag
    parser.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')

    return parser

def bool_flag(s):
    """ Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
