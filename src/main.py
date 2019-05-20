from __future__ import print_function
import argparse
import os
import sys
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--dlr', type=float, default=2e-4, help='learning rate for the discriminator, default=0.0002')
parser.add_argument('--glr', type=float, default=1e-3, help='learning rate for the generator and Q, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netQ', default='', help="path to netQ (to continue training)")
parser.add_argument('--outf', default='./checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--eval', type=int, default=-1, help='0 ~ (c_size-1)')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.MNIST(root='./mnist', download=True,
                    transform=transforms.Compose([
                        transforms.Resize(opt.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]))
nc=1
device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
c_size = 10
if opt.batchSize % c_size != 0:
    raise ValueError('batchSize must be divisible by c_size (%d)'%(c_size))


def saving_path(root_path):
    try:
        os.makedirs(root_path)
    except OSError:
        pass
    checkpoint_path = os.path.join(root_path, 'pth')
    result_path = os.path.join(root_path, 'result_img')
    try:
        os.makedirs(checkpoint_path)
        os.makedirs(result_path)
    except OSError:
        pass
    return checkpoint_path, result_path

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, c_size)
        )

    def forward(self, input, with_Q=False):
        output = self.main(input)
        if not with_Q:
            dis_out = self.discriminator(output)
            out = dis_out.view(-1, 1).squeeze(1)
            return out
        else:
            dis_out = self.discriminator(output)
            dis_out = dis_out.view(-1, 1).squeeze(1)
            # Q_out = self.Q(output.view(opt.batchSize, -1))
            return dis_out, output


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, c_size)
        )

    def forward(self, input, with_Q=False):
        # print(torch.mean(self.Q[0].weight.data))
        Q_out = self.Q(input.view(opt.batchSize, -1))
        return Q_out


def sample_noise(z_size, cond_cls, batch_size, fixed_noise=None):
    if fixed_noise is None:
        idx = np.random.randint(cond_cls, size=batch_size)
        cond = np.zeros((batch_size, cond_cls))
        cond[range(batch_size), idx] = 1.0
        cond = torch.Tensor(cond)

        noise = torch.randn(batch_size, z_size - cond_cls)
    else:
        idx = np.tile(np.arange(cond_cls), int(batch_size / cond_cls))
        one_hot = np.zeros((batch_size, cond_cls))
        one_hot[range(batch_size), idx] = 1.0
        cond = torch.Tensor(one_hot).view(batch_size, -1, 1, 1)

        noise = fixed_noise
        
    z = torch.cat([cond.to(device), noise.to(device)], 1).view(batch_size, -1, 1, 1)
    return z, idx


def trainIters():
    cp_path, img_path = saving_path(opt.outf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))
    netG = Generator().to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)
    netQ = Q().to(device)
    netQ.apply(weights_init)
    if opt.netQ != '':
        netQ.load_state_dict(torch.load(opt.netQ))
    # print(netQ)

    if opt.eval != -1:
        eval_result(opt, netG)
        return

    criterion_D = nn.BCELoss().to(device)
    criterion_Q = nn.CrossEntropyLoss().to(device)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam([{'params':netG.parameters()}, {'params':netQ.parameters()}], lr=opt.glr, betas=(opt.beta1, 0.999))

    mul = opt.batchSize // c_size
    fixed_noise = np.random.normal(size=(c_size, nz - c_size, 1, 1)).repeat(mul, axis=0)
    fixed_noise = torch.Tensor(fixed_noise, device=device)

    real_label = 1
    fake_label = 0
    tmp_dict = {
        'loss_d': [],
        'loss_g': [],
        'loss_q': [],
        'real_score': [],
        'fake_score_before': [],
        'fake_score_after': [],
    }
    history = tmp_dict.copy()

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader):
            # (1) Update D 
            # train with real
            optimizerD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion_D(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            z, c_idx = sample_noise(nz, c_size, batch_size)
            fake = netG(z)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion_D(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G & Q 
            optimizerG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            D_output, Q_input = netD(fake, with_Q=True)
            reconstruct_loss = criterion_D(D_output, label)
            D_G_z2 = D_output.mean().item()

            Q_output = netQ(Q_input)
            target = torch.LongTensor(c_idx).to(device)
            q_loss = criterion_Q(Q_output, target)

            G_Q_loss = reconstruct_loss + q_loss
            G_Q_loss.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader),
                    errD.item(), reconstruct_loss.item(), q_loss.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % img_path,
                        normalize=True, nrow=c_size)
                f_noise, _ = sample_noise(nz, c_size, batch_size, fixed_noise)
                netG.eval()
                fake = netG(f_noise)
                netG.train()
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (img_path, epoch),
                        normalize=True, nrow=c_size)
                save_history(tmp_dict, errD.item(), reconstruct_loss.item(), q_loss.item(), D_x, D_G_z1, D_G_z2, history)
            else:
                save_history(tmp_dict, errD.item(), reconstruct_loss.item(), q_loss.item(), D_x, D_G_z1, D_G_z2)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (cp_path, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (cp_path, epoch))
        torch.save(netQ.state_dict(), '%s/netQ_epoch_%d.pth' % (cp_path, epoch))


def save_history(tmp_dict, loss_d, loss_g, loss_q, D_x, D_G_z1, D_G_z2, save_dict=None):
    tmp_dict['loss_d'].append(loss_d)
    tmp_dict['loss_g'].append(loss_g)
    tmp_dict['loss_q'].append(loss_q)
    tmp_dict['real_score'].append(D_x)
    tmp_dict['fake_score_before'].append(D_G_z1)
    tmp_dict['fake_score_after'].append(D_G_z2)
    if save_dict is not None:
        for key,val in tmp_dict.items():
            save_dict[key].append(float(np.array(val).mean()))
            tmp_dict[key] = []
        with open(os.path.join(opt.outf, 'history.json'), 'w') as f:
            json.dump(save_dict, f)


def eval_result(opt, G):
    if opt.eval >= 0 and opt.eval < c_size:
        one_hot = np.zeros(c_size)
        one_hot[opt.eval] = 1.0
        cond = torch.Tensor(one_hot, device=device).view(1, -1, 1, 1)
        
        fixed_noise = np.random.normal(size=(1, nz - c_size, 1, 1))
        fixed_noise = torch.Tensor(fixed_noise, device=device)
        z = torch.cat([cond, fixed_noise], 1).view(1, -1, 1, 1)
        
        G.eval()
        fake = G(z)
        G.train()
        vutils.save_image(fake.detach(),
                '%s/fake_eval_%d.png' % ('./', opt.eval),
                normalize=True)
    else:
        raise ValueError('eval number must between 0 and c_size-1')


if __name__ == '__main__':
    trainIters()
    sys.stdout.flush()
    sys.exit()