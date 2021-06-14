import argparse
from trainer import train
import torch

from utils.config import get_config
from utils.eprint import eprint

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--batch', type=int, default=0)
parser.add_argument('-l', '--lambda_value', type=float, default=None)
parser.add_argument('-t', '--temperature', type=float, default=None)
args = parser.parse_args()

config_path = args.config
config = get_config(config_path)
if args.temperature is not None:
    config.train.scl.temperature = args.temperature
if args.lambda_value is not None:
    config.train.scl.lambda_value = args.lambda_value
if args.batch == 0:
    batch_size = config.train.batch_size
    batch_size *= torch.cuda.get_device_properties(0).total_memory // 8370061312
    eprint("BATCH SIZE CHANGE FROM {} to {}".format(config.train.batch_size, batch_size))
    config.train.batch_size = batch_size
else:
    eprint("BATCH SIZE CHANGE FROM {} to {}".format(config.train.batch_size, args.batch))
    config.train.batch_size = args.batch
train(config)