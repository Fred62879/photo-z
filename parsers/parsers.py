
import sys
import yaml
import pprint
import logging
import argparse


def default_log_setup(level=logging.INFO):
    """ Sets up default logging, always logging to stdout.
        :param level: logging level, e.g. logging.INFO
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    # TODO: better to also use loggers per file and add %(name)15s
    logging.basicConfig(level=level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)

def parse_input_config(parser):
    args = parser.parse_args()

    # parse yaml config file
    if hasattr(args, 'config') and args.config is not None:
        parse_yaml_config(args.config, parser)
    args = parser.parse_args()

    config = vars(args)
    args_str = argparse_to_str(parser, args)
    return config, args_str

def argparse_to_str(parser, args):
    """ Convert parser to string representation for Tensorboard logging.
        Args:
          args : The parsed arguments.
        Returns:
          arg_str : The string to be printed.
    """
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))
    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f"```{args_str}```"
    return args_str

def parse_yaml_config(config_path, parser):
    """ Parses and sets the parser defaults with a yaml config file.
        Args:
          config_path : path to the yaml config file.
          parser : The parser for which the defaults will be set.
          parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)

    defaults_dict = {}

    # Load the parent config if it exists
    parent_config_path = config_dict.pop("parent", None)

    if parent_config_path is not None:
        if not os.path.isabs(parent_config_path):
            parent_config_path = os.path.join(os.path.split(config_path)[0], parent_config_path)
        with open(parent_config_path) as f:
            parent_config_dict = yaml.safe_load(f)
        if "parent" in parent_config_dict.keys():
            raise Exception("Hierarchical configs of more than 1 level deep are not allowed.")
        for key in parent_config_dict:
            for field in parent_config_dict[key]:
                if field not in list_of_valid_fields:
                    raise ValueError(
                        f"ERROR: {field} is not a valid option. Check for typos in the config."
                    )
                defaults_dict[field] = parent_config_dict[key][field]

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren"t actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]

    parser.set_defaults(**defaults_dict)
