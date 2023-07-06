import numpy as np
import healpy as hp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from .model import Generator, Discriminator, compute_avg_mat
from tqdm.auto import tqdm
from copy import deepcopy


class ArrayDataset(Dataset):
    def __init__(self, path: str, kappa_std):
        std = kappa_std.astype(np.float32)[None, :, None]
        self.data = np.load(path).astype(np.float32) / std
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    def __init__(self, num_bins, gen_mask, disc_mask, real_data, fake_data, gen=None, disc=None, 
                 num_channels=8, ident_loss_hp=0.1, scale_loss_hp=1., batch_size=128,
                 gen_opt=None, disc_opt=None, device=None, save_name=None, save_every=1000, writer_dir=None):
        self.num_bins = num_bins
        self.nside = hp.get_nside(gen_mask)
        self.gen_mask = gen_mask
        self.disc_mask = disc_mask
        self.disc_mask_in_gen_mask = self.disc_mask[self.gen_mask]
        self.real_data = real_data
        self.fake_data = fake_data
        self.gen = gen
        self.disc = disc
        self.num_channels  = num_channels
        self.batch_size = 128
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.device = device
        
        self.save_name = save_name
        self.save_every = save_every
        self.writer = SummaryWriter(writer_dir) if writer_dir else None
        self.i = 0

        if not self.device:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        if not self.gen:
            avg_mat = compute_avg_mat(self.nside, self.gen_mask).to(self.device)
            self.gen = Generator(self.num_bins, avg_mat, num_channels=self.num_channels).to(self.device)

        if not self.disc:
            avg_mat = compute_avg_mat(self.nside, self.disc_mask).to(self.device)
            self.disc = Discriminator(self.num_bins, avg_mat, num_channels=self.num_channels).to(self.device)

        if not self.gen_opt:
            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=1e-3, betas=(0., 0.99))

        if not self.disc_opt:
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=4e-3, betas=(0., 0.99))

        self.ident_loss_hp = ident_loss_hp * torch.tensor(np.ones(self.num_bins)[None,:,None], requires_grad=False).to(self.device)
        self.scale_loss_hp = scale_loss_hp * torch.tensor(np.ones(self.num_bins)[None,:], requires_grad=False).to(self.device)

        self.gen_ema = deepcopy(self.gen)

        self.real_loader = DataLoader(real_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
        self.fake_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

        self.real_iter = iter(self.real_loader)
        self.fake_iter = iter(self.fake_loader)

    def save_models(self, fname):
        ckpt = {'iter': self.i,
                'gen': self.gen.state_dict(),
                'disc': self.disc.state_dict(),
                'gen_ema': self.gen_ema.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'disc_opt': self.disc_opt.state_dict()}

        torch.save(ckpt, fname)

    def load_models(self, fname):
        ckpt = torch.load(fname)

        self.i = ckpt['iter']
        self.gen = ckpt['gen']
        self.disc = ckpt['disc']
        self.gen_ema = ckpt['gen_ema']
        self.gen_opt = ckpt['gen_opt']
        self.disc_opt = ckpt['disc_opt']

    @torch.no_grad()
    def update_gen_ema(self, alpha=0.999):
        for p, ema_p in zip(self.gen.parameters(), self.gen_ema.parameters()):
            ema_p.data.mul_(alpha).add_(1 - alpha, p.data)

    def train_disc(self):
        try:
            real = next(self.real_iter).to(self.device)
        except:
            self.real_iter = iter(self.real_loader)
            real = next(self.real_iter).to(self.device)

        try:
            fake = next(self.fake_iter).to(self.device)
        except:
            self.fake_iter = iter(self.fake_loader)
            fake = next(self.fake_iter).to(self.device)

        real.requires_grad_(True)

        real_out = self.disc(real[..., self.disc_mask_in_gen_mask])
        real_loss = F.relu(1 - real_out).mean()
        loss = real_loss
        loss.backward()

        with torch.no_grad():
            fake = self.gen(fake)
        fake_out = self.disc(fake[..., self.disc_mask_in_gen_mask])
        fake_loss = F.relu(1 + fake_out).mean()
        fake_loss.backward()

        return real_loss, fake_loss

    def train_gen(self):
        try:
            fake = next(self.fake_iter).to(self.device)
        except:
            self.fake_iter = iter(self.fake_loader)
            fake = next(self.fake_iter).to(self.device)

        gen_out = self.gen(fake)
        fake_out = self.disc(gen_out[..., self.disc_mask_in_gen_mask])
        fake_loss = -fake_out.mean()
        ident_loss = ((gen_out - fake).square() * self.ident_loss_hp).mean()
        scale_loss = ((gen_out - fake).mean(dim=-1).square() * self.scale_loss_hp).mean()
        loss = fake_loss + ident_loss + scale_loss
        loss.backward()

        return fake_loss.item(), ident_loss.item(), scale_loss.item()

    def train(self, num_iter):
        for _ in tqdm(range(num_iter)):
            self.disc.requires_grad_(True)
            self.disc_opt.zero_grad()
            disc_loss = self.train_disc()
            self.disc_opt.step()

            self.disc.requires_grad_(False)
            self.gen_opt.zero_grad()
            gen_loss = self.train_gen()
            self.gen_opt.step()
            self.update_gen_ema()

            if self.writer:
                self.writer.add_scalars('disc', {'real': disc_loss[0], 'fake': disc_loss[1]}, self.i)
                self.writer.add_scalars('gen', {'disc': gen_loss[0], 'ident': gen_loss[1], 'scale': gen_loss[2]},
                                        self.i)

            if self.save_name and self.i % self.save_every == 0:
                self.save_models(self.save_name)

            self.i += 1

        if self.save_name:
            self.save_models(self.save_name)
