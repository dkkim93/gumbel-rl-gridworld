import torch
import gym
import yaml
import git
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters

    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to load config from. Default: "."
    """
    with open(path + "/config/" + args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        args.__dict__[key] = value


def set_logger(logger_name, log_file, level=logging.INFO):
    """Sets python logging

    Args:
        logger_name (str): Specifies logging name
        log_file (str): Specifies path to save logging
        level (int): Logging when above specified level. Default: logging.INFO
    """
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args, path="."):
    """Loads and replaces default parameters with experiment
    specific parameters

    Args:
        args (argparse): Python argparse that contains arguments
        path (str): Root directory to get Git repository. Default: "."

    Examples:
        log[args.log_name].info("Hello {}".format("world"))

    Returns:
        log (dict): Dictionary that contains python logging
    """
    log = {}
    set_logger(
        logger_name=args.log_name,
        log_file=r'{0}{1}'.format("./log/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    for arg, value in sorted(vars(args).items()):
        log[args.log_name].info("%s: %r", arg, value)

    repo = git.Repo(path)
    log[args.log_name].info("Branch: {}".format(repo.active_branch))
    log[args.log_name].info("Commit: {}".format(repo.head.commit))

    return log


def make_env(args):
    import gym_env  # noqa
    env = gym.make(
        args.env_name, row=args.row, col=args.col, n_action=args.n_action)
    env._max_episode_steps = args.ep_max_timesteps

    return env


def onehot_from_logits(logits):
    """Given batch of logits, return one-hot sample
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Add a dimension

    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if device == torch.device("cuda"):
        tens_type = torch.cuda.FloatTensor
    elif device == torch.device("cpu"):
        tens_type = torch.FloatTensor
    else:
        raise ValueError("Invalid dtype")

    y = logits + sample_gumbel(logits.shape, tens_type=tens_type)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Add a dimension
    assert len(logits.shape) == 2, "Shape should be: (# of batch, # of action)"

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y

    return y
