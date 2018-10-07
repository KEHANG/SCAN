"""main.py"""

import argparse
import os

import numpy as np
import torch

from solver import ori_beta_VAE, DAE, beta_VAE, SCAN
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--SCAN', action='store_true', help='whether to train a SCAN model or the original beta-VAE model')
parser.add_argument('--phase', default='DAE', type=str, help='the stage of the training, which has 3 stages: {DAE, beta-VAE, SCAN}')

parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')

parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

parser.add_argument('--vis_on', default=True, type=str2bool, help='enable visdom visualization')
parser.add_argument('--vis_port', default=6059, type=str, help='visdom port number')
parser.add_argument('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
parser.add_argument('--display_save_step', default=10000, type=int, help='number of iterations after which to display data and save checkpoint')

parser.add_argument('--root_dir', default='/data/hc/SCAN', type=str, help='root directory')
parser.add_argument('--env_name', default='trial', type=str, help='visdom env name')
parser.add_argument('--ref_env_name', default='', type=str, help='directory of the reference model')
parser.add_argument('--dset_dir', default='dataset', type=str, help='dataset directory')
parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_name', default='last', type=str, help='name of the previous checkpoint')

args = parser.parse_args()

args.dset_dir = os.path.join(args.root_dir, args.dset_dir)
args.output_dir = os.path.join(args.root_dir, args.output_dir, args.env_name)
args.ckpt_dir = os.path.join(args.root_dir, args.ckpt_dir, args.env_name)

args.cuda = args.cuda and torch.cuda.is_available()

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if not args.SCAN:
        model = ori_beta_VAE
    else:
        if args.phase == 'DAE':
            model = DAE
        elif args.phase == 'beta_VAE':
            model = beta_VAE
        elif args.phase == 'SCAN':
            model = SCAN
    model = model(args)

    if args.train:
        model.train()
    else:
        model.vis_traverse()

if __name__ == "__main__":
    main(args)
