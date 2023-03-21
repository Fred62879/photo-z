
import torch
import logging
import argparse

from parsers.parsers import parse_input_config, default_log_setup

str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

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
    parser = argparse.ArgumentParser(
        description="ArgumentParser for protei binding factor prediction.")

    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group("global")

    global_group.add_argument("--config", type=str, help="Path to config file to replace defaults.")

    global_group.add_argument("--use_gpu", action="store_true")
    global_group.add_argument("--verbose", action="store_true")
    global_group.add_argument("--print-shape", action="store_true")
    global_group.add_argument("--activate_timer", action="store_true")
    global_group.add_argument("--exp-name", type=str, help="Experiment name.")

    ###################
    # General global network things
    ###################
    net_group = parser.add_argument_group("net")

    net_group.add_argument("--d-in", type=int)
    net_group.add_argument("--d-hidden", type=int)
    net_group.add_argument("--d-out", type=int)
    net_group.add_argument("--n-layers", type=int)
    net_group.add_argument("--skip-in", type=int)
    net_group.add_argument("--multires", type=int)
    net_group.add_argument("--bias", type=int)
    net_group.add_argument("--scale", type=int)
    net_group.add_argument("--geometric-init", action="store_true")
    net_group.add_argument("--weight-norm", action="store_true")

    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group("dataset")

    data_group.add_argument("--shuffle-dataloader", action="store_true")
    data_group.add_argument("--dataloader-drop-last", action="store_true")
    data_group.add_argument("--dataloader-num-workers", type=int, default=0)

    data_group.add_argument("--data-path", type=str, help="Path to the dataset")
    data_group.add_argument("--redshift-fname", type=str, help="Filename of source redshift.")
    data_group.add_argument("--dataset-num-workers", type=int, default=-1,
                            help="Number of workers for dataset preprocessing, if it \
                            supports multiprocessing. -1 indicates no multiprocessing.")

    ###################
    # Arguments for optimizer
    ###################

    optim_group = parser.add_argument_group("optimizer")

    optim_group.add_argument("--optimizer-type", type=str, default="adam", choices=list(str2optim.keys()),
                             help="Optimizer to be used.")
    optim_group.add_argument("--lr", type=float, default=0.001,
                             help="Learning rate.")
    optim_group.add_argument("--b1",type=float, default=0.5)
    optim_group.add_argument("--b2",type=float, default=0.999)

    ###################
    # Arguments for training
    ###################

    train_group = parser.add_argument_group("trainer")

    train_group.add_argument("--trainer-mode", type=str)

    train_group.add_argument("--num-epochs", type=int, default=250,
                             help="Number of epochs to run the training.")

    train_group.add_argument("--warm-up-end", type=int, default=1000)

    train_group.add_argument("--batch-size", type=int, default=512,
                             help="Batch size for the training.")
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
    train_group.add_argument("--pretrained-model-name", type=str)

    train_group.add_argument("--log-dir", type=str, default="_results/logs/runs/",
                             help="Log file directory for checkpoints.")

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
