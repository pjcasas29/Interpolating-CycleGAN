#!/usr/bin/python3

from __future__ import unicode_literals

import argparse
import itertools
import sys
import shutil
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn 
import os
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import tensor2image
from utils import weights_init_normal
from datasets import ImageDataset
from tensorboardX import SummaryWriter
from prompt_toolkit import prompt

#os.environ['CUDA_VISIBLE_DEVICES']='2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='', help='Experiment name')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_epoch', type=int, default=10, help='number of epoch multiple which to save models')

opt = parser.parse_args()
print(opt)

if opt.experiment == '':
    print("ERROR: Must provide experiment name for logging")
    sys.exit()

output_path = 'experiments/' + opt.experiment + '/'

if os.path.exists(output_path):
    text = prompt("Experiment directory already exists. Would you like to overwrite? (y/N): ")
    if text is '' or text is 'N' or text is 'n':
        sys.exit()
    elif text is 'y' or text is 'Y':
        shutil.rmtree(output_path)

os.mkdir(output_path)
os.mkdir(output_path + '/net_checkpoints/')
writer = SummaryWriter(output_path)

log = open(output_path + 'params', "w+")
log.write(str(opt)) 
log.close()    

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = nn.DataParallel(Generator(opt.input_nc, opt.output_nc))
netG_B2A = nn.DataParallel(Generator(opt.output_nc, opt.input_nc))
netD_A = nn.DataParallel(Discriminator(opt.input_nc))
netD_B1 = nn.DataParallel(Discriminator(opt.output_nc))
netD_B2 = nn.DataParallel(Discriminator(opt.output_nc))

#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B1.cuda()
    netD_B2.cuda()
"""
if opt.cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B1.to(device)
    netD_B2.cuda()
"""
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B1.apply(weights_init_normal)
netD_B2.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B1 = torch.optim.Adam(netD_B1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B2 = torch.optim.Adam(netD_B2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(output_path, opt.n_epochs, len(dataloader), opt.batchSize)

###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    
    for i, batch in enumerate(dataloader):
        if len(batch['A']) < opt.batchSize:
            print("batch length: " + str(len(batch['A'])) + " batchSize: " + str(opt.batchSize))
            continue
        # Set model input
        
        real_A = Variable(input_A.copy_(batch['A']))

        sample = 'B1' if i % 2 == 0 else 'B2'
        real_B = Variable(input_B.copy_(batch[sample]))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B1(fake_B) if i % 2 == 0 else netD_B2(fake_B) 
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()
 
       # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        
        loss_D_B1 = torch.tensor(1.0)
        loss_D_B2 = torch.tensor(1.0)

         
        if i % 2 ==0:

            optimizer_D_B1.zero_grad()


            # Real loss
            pred_real = netD_B1(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
        
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B1(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss

            loss_D_B1 = (loss_D_real + loss_D_fake)*0.5
            loss_D_B1.backward()

            optimizer_D_B1.step()
            
            global log_real_A1
            global log_real_B1
            global log_fake_A1
            global log_fake_B1
            
            log_real_A1 = real_A.clone()
            log_real_B1 = real_B.clone()
            log_fake_A1 = fake_A.clone()
            log_fake_B1 = fake_B.clone()
            

            logger.log(losses={'loss_G': loss_G, 'loss_G_identity_A': loss_identity_A,'loss_G_identity_B': loss_identity_B, 'loss_G_identity': (loss_identity_A + loss_identity_B),'loss_G_GAN_A2B': loss_GAN_A2B,'loss_G_GAN_B2A': loss_GAN_B2A, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_cycle_BAB': loss_cycle_BAB, 'loss_cycle_ABA': loss_cycle_ABA, 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_DA':loss_D_A, 'loss_DB1': loss_D_B1, 'loss_DB2': loss_D_B2}, images={})#'real_A1': log_real_A1, 'real_B1': log_real_B1, 'fake_A1': log_fake_A1, 'fake_B1': log_fake_B1})
    
        else:

            optimizer_D_B2.zero_grad()

            # Real loss
            pred_real = netD_B2(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
        
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B2(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss

            loss_D_B2 = (loss_D_real + loss_D_fake)*0.5
            loss_D_B2.backward()

            optimizer_D_B2.step()

            global log_real_A2
            global log_real_B2
            global log_fake_A2
            global log_fake_B2

            log_real_A2 = real_A.clone()
            log_real_B2 = real_B.clone()
            log_fake_A2 = fake_A.clone()
            log_fake_B2 = fake_B.clone()

            logger.log(losses={'loss_G': loss_G, 'loss_G_identity_A': loss_identity_A,'loss_G_identity_B': loss_identity_B, 'loss_G_identity': (loss_identity_A + loss_identity_B),'loss_G_GAN_A2B': loss_GAN_A2B,'loss_G_GAN_B2A': loss_GAN_B2A, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_cycle_BAB': loss_cycle_BAB, 'loss_cycle_ABA': loss_cycle_ABA, 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_DA':loss_D_A, 'loss_DB1': loss_D_B1, 'loss_DB2': loss_D_B2}, images={})#'real_A2': log_real_A2, 'real_B2': log_real_B2, 'fake_A2': log_fake_A2, 'fake_B2': log_fake_B2})
                
        ###################################
        
        # Progress report (http://localhost:8097)
        


    #writer.add_scalar('loss_G', loss_G.item(), epoch)
    #writer.add_scalar('loss_G_identity_A', loss_identity_A.item(), epoch)
    #writer.add_scalar('loss_G_identity_B', loss_identity_B.item(), epoch)
    #writer.add_scalar('loss_G_identity', (loss_identity_A+loss_identity_B).item(), epoch)
    #writer.add_scalar('loss_G_GAN_A2B', loss_GAN_A2B.item(), epoch)
    #writer.add_scalar('loss_G_GAN_B2A', loss_GAN_B2A.item(), epoch)
    #writer.add_scalar('loss_G_GAN', (loss_GAN_A2B + loss_GAN_B2A).item(), epoch)
    #writer.add_scalar('loss_cycle_BAB', loss_cycle_BAB.item(), epoch)
    #writer.add_scalar('loss_cycle_ABA', loss_cycle_ABA.item(), epoch)
    #writer.add_scalar('loss_cycle', (loss_cycle_ABA + loss_cycle_BAB).item(), epoch)
    #writer.add_scalar('loss_D_A', loss_D_A.item(), epoch)
    #writer.add_scalar('loss_D_B1', loss_D_B1.item(), epoch)
    #writer.add_scalar('loss_D_B2', loss_D_B2.item(), epoch)
    #[writer.add_image('real_A1', log_real_A1[i,:,:,:], epoch) for i in range(0, len(batch['A']))]
    #[writer.add_image('real_B1', log_real_B1[i,:,:,:], epoch) for i in range(0, len(batch['B1']))]
    #[writer.add_image('fake_A1', log_fake_A1[i,:,:,:], epoch) for i in range(0, len(batch['A']))]
    #[writer.add_image('fake_B1', log_fake_B1[i,:,:,:], epoch) for i in range(0, len(batch['B1']))]
    #[writer.add_image('real_A2', log_real_A2[i,:,:,:], epoch) for i in range(0, len(batch['A']))]
    #[writer.add_image('real_B2', log_real_B2[i,:,:,:], epoch) for i in range(0, len(batch['B2']))]
    #[writer.add_image('fake_A2', log_fake_A2[i,:,:,:], epoch) for i in range(0, len(batch['A']))]
    #[writer.add_image('fake_B2', log_fake_B2[i,:,:,:], epoch) for i in range(0, len(batch['B2']))]
    
    logger.log_images(losses={}, images={'real_A2': log_real_A2, 'real_B2': log_real_B2, 'fake_A2': log_fake_A2, 'fake_B2': log_fake_B2, 'real_A1': log_real_A1, 'real_B1': log_real_B1, 'fake_A1': log_fake_A1, 'fake_B1': log_fake_B1})
    
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B1.step()
    lr_scheduler_D_B2.step()

    torch.save(netG_A2B.state_dict(), output_path + 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), output_path + 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), output_path + 'netD_A.pth')
    torch.save(netD_B1.state_dict(), output_path + 'netD_B1.pth')
    torch.save(netD_B2.state_dict(), output_path + 'netD_B2.pth')

    if epoch % opt.save_epoch == 0:
        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), output_path + '/net_checkpoints/netG_A2B_epoch' + str(epoch) + '.pth')
        torch.save(netG_B2A.state_dict(), output_path + '/net_checkpoints/netG_B2A_epoch' + str(epoch) + '.pth')
        torch.save(netD_A.state_dict(), output_path + '/net_checkpoints/netD_A_epoch' + str(epoch) + '.pth')
        torch.save(netD_B1.state_dict(), output_path + '/net_checkpoints/netD_B1_epoch' + str(epoch) + '.pth')
        torch.save(netD_B2.state_dict(), output_path + '/net_checkpoints/netD_B2_epoch' + str(epoch) + '.pth')
        
###################################

