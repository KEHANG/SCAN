"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from abc import ABC, abstractmethod
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model import BetaVAE_H_net, BetaVAE_B_net, DAE_net, SCAN_net
from dataset import return_data

#---------------------------------NEW CLASS-------------------------------------#
class Solver(ABC):
    def __init__(self, args, model):
        self.global_iter = 0
        self.args = args

        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if not os.path.exists(self.args.ckpt_dir):
            os.makedirs(self.args.ckpt_dir, exist_ok=True)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.vis_on:
            self.vis = visdom.Visdom(port=self.args.vis_port)
        self.gather = DataGather()
        self.net = cuda(model(self.args.z_dim, self.nc), self.args.cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.args.lr,
                               betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon)
        self.load_checkpoint(self.args.ckpt_name)
        self.data_loader = return_data(self.args)


    @abstractmethod
    def prepare_training(self):
        pass
    @abstractmethod
    def training_process(self, x):
        pass
    @abstractmethod
    def get_win_states(self):
        pass
    @abstractmethod
    def load_win_states(self):
        pass

    def train(self):
        self.net_mode(train=True)
        self.prepare_training()

        self.pbar = tqdm(total=self.args.max_iter)
        self.pbar.update(self.global_iter)
        while self.global_iter < self.args.max_iter:
            for x in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x = Variable(cuda(x, self.args.cuda))
                loss = self.training_process(x)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.global_iter%self.args.display_save_step == 0:
                    self.save_checkpoint(self.get_win_states(), str(self.global_iter))
                    self.save_checkpoint(self.get_win_states(), 'last')
                    self.pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def vis_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.vis.images(images, env=self.args.env_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)
    def vis(self, x, x_recon, traverse=True):
        if self.args.vis_on:
            self.gather.insert(images=x.data)
            self.gather.insert(images=F.sigmoid(x_recon).data)
            self.vis_reconstruction()
            self.vis_lines()
            self.gather.flush()

        if (self.args.vis_on or self.args.save_output) and traverse:
            self.vis_traverse()

    def update_win(self, Y, win, legend, title):
        iters =  torch.Tensor(self.gather.data['iter'])
        opts = dict( width=400, height=400, legend=legend, xlabel='iteration', title=title,)
        if win is None:
            return self.vis.line(X=iters, Y=Y, env=self.args.env_name+'_lines', opts=opts)
        else:
            return self.vis.line(X=iters, Y=Y, env=self.args.env_name+'_lines', win=win, update='append', opts=opts)
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')
        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, win_states, filename, silent=True):
        states = {'iter': self.global_iter,
                  'win_states': win_states,
                  'net_states': self.net.state_dict(),
                  'optim_states': self.optim.state_dict(),}

        file_path = os.path.join(self.args.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
    def load_checkpoint(self, filename):
        file_path = os.path.join(self.args.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.load_win_states(checkpoint['win_states'])
            self.net.load_state_dict(checkpoint['net_states'])
            self.optim.load_state_dict(checkpoint['optim_states'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


#---------------------------------NEW CLASS-------------------------------------#
class super_beta_VAE(Solver):
    def __init__(self, args):
        if args.model == 'H':
            model = BetaVAE_H_net
        elif args.model == 'B':
            model = BetaVAE_B_net
        else:
            raise NotImplementedError('only support model H or B')
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        super(super_beta_VAE, self).__init__(args, model)

    def prepare_training(self):
        self.args.C_max = Variable(cuda(torch.FloatTensor([self.args.C_max]), self.args.cuda))
    def training_process(self, x):
        x_recon, mu, logvar = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        if self.args.objective == 'H':
            loss = recon_loss + self.args.beta*total_kld
        elif self.args.objective == 'B':
            C = torch.clamp(self.args.C_max/self.args.C_stop_iter*self.global_iter, 0, self.args.C_max.data[0])
            loss = recon_loss + self.args.gamma*(total_kld-C).abs()

        if self.args.vis_on and self.global_iter % self.args.gather_step == 0:
            self.gather.insert(iter=self.global_iter,
                               mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                               recon_loss=recon_loss.data, total_kld=total_kld.data,
                               dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

        if self.global_iter % self.args.display_save_step == 0:
            self.pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                self.global_iter, recon_loss.data[0], total_kld.data[0], mean_kld.data[0]))

            var = logvar.exp().mean(0).data
            var_str = ''
            for j, var_j in enumerate(var):
                var_str += 'var{}:{:.4f} '.format(j+1, var_j)
            self.pbar.write(var_str)

            if self.args.objective == 'B':
                self.pbar.write('C:{:.3f}'.format(C.data[0]))

            self.vis(x, x_recon)

        return loss

    def vis_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        variances = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()

        legend = []
        for z_j in range(self.args.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        self.win_recon = self.update_win(recon_losses, self.win_recon, ['reconstruction loss'], 'reconstruction loss')
        self.win_kld = self.update_win(klds, self.win_kld, legend, 'kl divergence')
        self.win_mu = self.update_win(mus, self.win_mu, legend[:self.args.z_dim], 'posterior mean')
        self.win_var = self.update_win(variances, self.win_var, legend[:self.args.z_dim], 'posterior variance')

        self.net_mode(train=True)
    def vis_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.args.cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.args.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.args.z_dim), self.args.cuda), volatile=True)

        if self.args.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.args.cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.args.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.args.cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.args.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.args.cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.args.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.args.cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.args.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.args.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            if self.args.vis_on:
                self.vis.images(samples, env=self.args.env_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.args.save_output:
            output_dir = os.path.join(self.args.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.args.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.args.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def get_win_states(self):
        return {'recon': self.win_recon,
                'kld': self.win_kld,
                'mu': self.mu,
                'var': self.var,}
    def load_win_states(self, win_states):
        self.win_recon = win_states['recon']
        self.win_kld = win_states['kld']
        self.win_var = win_states['var']
        self.win_mu = win_states['mu']


#---------------------------------NEW CLASS-------------------------------------#
class ori_beta_VAE(super_beta_VAE):
    def __init__(self, args):
        super(ori_beta_VAE, self).__init__(args)


#---------------------------------NEW CLASS-------------------------------------#
class beta_VAE(super_beta_VAE):
    def __init__(self, args):
        pass

    def load_DAE_checkpoint(self):
        pass

#---------------------------------NEW CLASS-------------------------------------#
class DAE(Solver):
    def __init__(self, args):
        self.win_recon = None
        super(Solver, self).__init__(args, DAE_net)

    def prepare_training(self):
        pass
    def training_process(self, x):

        x_recon = self.net(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        loss = recon_loss

        if self.args.vis_on and self.global_iter % self.args.gather_step == 0:
            self.gather.insert(iter=self.global_iter, recon_loss=recon_loss.data)
        if self.global_iter % self.args.display_save_step == 0:
            self.pbar.write('[{}] recon_loss:{:.3f}'.format(self.global_iter, recon_loss.data[0]))
            self.vis(x, x_recon, traverse=False)

        return loss

    def get_win_states(self):
        return {'recon': self.win_recon}
    def load_win_states(self, win_states):
        self.win_recon = win_states['recon']

    def vis_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        self.win_recon = self.update_win(recon_losses, self.win_recon, ['reconstruction loss'], 'reconstruction loss')
        self.net_mode(train=True)

#---------------------------------NEW CLASS-------------------------------------#
class SCAN(Solver):
    def __init__(self, args):
        super(SCAN, self).__init__(args)

        self.set_net_and_optim(SCAN_net)


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None
    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
