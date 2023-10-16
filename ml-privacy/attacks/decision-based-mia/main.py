import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import time
import random
import time
import math
import numpy as np
from runx.logx import logx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from models import ResNet18
from classifier import CNN
from utils import load_dataset, init_func, Rand_Augment
from deeplearning import train_target_model, test_target_model, train_shadow_model, test_shadow_model
from attack import AdversaryTwo_HopSkipJump, AdversaryOne_evaluation, AdversaryOne_Feature
from cert_radius.certify import certify


action = -1
# device_alt = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_alt = "cuda:0"
def Train_Target_Model(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode=args.mode_type)
        targetmodel = CNN('CNN7', dataset)
        targetmodel.apply(init_func)
        if (device_alt == "cpu"):
            pass
            # targetmodel = nn.DataParallel(targetmodel)
        else:
            targetmodel = nn.DataParallel(targetmodel.cuda())
        optimizer = optim.Adam(targetmodel.parameters(), lr=args.lr)
        logx.msg('======================Train_Target_Model {} ===================='.format(cluster))
        for epoch in range(1, args.epochs + 1):
            train_target_model(args, targetmodel, train_loader, optimizer, epoch)
            test_target_model(args, targetmodel, test_loader, epoch, save=True)

            
def AdversaryTwo(args, Random_Data=False):
    if Random_Data:
        args.Split_Size = [[100], [2000], [100], [100]]
        img_sizes = [(3,32,32), (3,32,32), (3,64,64), (3, 128, 128)] 
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    num_class = args.num_classes[args.dataset_ID]
    
    logx.initialize(logdir=args.logdir + '/adversaryTwo', coolname=False, tensorboard=False)
    if args.blackadvattack == 'HopSkipJump':
        ITER = [50] # for call HSJA evaluation [1, 5, 10, 15, 20, 30]  default 50
    for maxitr in ITER:
        AUC_Dist, Distance = [], []
        for cluster in split_size:
            torch.cuda.empty_cache()
            args.batch_size = 1
            if Random_Data:
                fake_set = datasets.FakeData(size=10000, image_size=img_sizes[args.dataset_ID], num_classes=num_class, transform= transforms.Compose([Rand_Augment(), transforms.ToTensor()]))
                data_loader = DataLoader(fake_set, batch_size=args.batch_size, shuffle=False)
            else:
                data_loader = load_dataset(args, dataset, cluster, mode='adversary', max_num=200)
            targetmodel = CNN('CNN7', dataset)
            if (device_alt == "cpu"):
                pass
                # targetmodel = nn.DataParallel(targetmodel)
            else:
                targetmodel = nn.DataParallel(targetmodel.cuda())
            
            state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
            targetmodel.load_state_dict(state_dict)
            targetmodel.eval()
            
            if args.blackadvattack == 'HopSkipJump':
                AUC_Dist, Distance = AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, Random_Data, maxitr)

        df = pd.DataFrame()
        AUC_Dist = df.append(AUC_Dist, ignore_index=True)
        Distance = df.append(Distance, ignore_index=True)
        
        if Random_Data:
            AUC_Dist.to_csv(args.logdir + '/adversaryTwo/AUC_Dist_'+args.blackadvattack+'.csv')
            Distance.to_csv(args.logdir + '/adversaryTwo/Distance_Random_'+args.blackadvattack+'.csv')
        else:
            AUC_Dist.to_csv(args.logdir + '/adversaryTwo/AUC_Dist_'+args.blackadvattack + '.csv')
            Distance.to_csv(args.logdir + '/adversaryTwo/Distance_'+args.blackadvattack+'.csv')
        

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 
    parser.add_argument('--action', default=0, type=int, help='train or attack')    
    parser.add_argument('--train', default=True, type=bool, help='train or attack')
    parser.add_argument('--dataset_ID', default=False, type=int, help='CIFAR10=0, CIFAR100=1')
    parser.add_argument('--datasets', nargs='+', default=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--num_classes', nargs='+', default=[10, 100])
    parser.add_argument('--Split-Size', nargs='+', default=[[3000],[7000]]) 
    parser.add_argument('--batch-size', nargs='+', default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', default=True,type=bool, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--blackadvattack', default='HopSkipJump', type=str, help='adversaryTwo uses the adv attack the target Model: HopSkipJump')
    parser.add_argument('--logdir', type=str, default='', help='target log directory')
    parser.add_argument('--mode_type', type=str, default='', help='the type of action referring to the load dataset')
    parser.add_argument('--advOne_metric', type=str, default='Loss_visual', help='AUC of Loss, Entropy, Maximum respectively; or Loss_visual')
    args = parser.parse_args()

    for dataset_idx in [0,1]:
        args.dataset_ID = dataset_idx
        args.logdir = 'results'+'/' + args.datasets[args.dataset_ID]
        action = 1
        
        # train
        if action == 0:
            args.mode_type = 'target'
            Train_Target_Model(args)

        # attack
        elif action == 1:
            AdversaryTwo(args, Random_Data=False)
            
            
if __name__ == "__main__":
    main()
