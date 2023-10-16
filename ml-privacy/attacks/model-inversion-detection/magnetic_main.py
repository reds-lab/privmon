import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import torch
from classify import *
from generator import *
from discri import *
from torch.nn import DataParallel
import time
import random
import os, logging
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import yaml
from magnetic_attack import magnetic_attack
from art.estimators.classification.pytorch import PyTorchClassifier
import argparse

# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    global args, logger

    parser = argparse.ArgumentParser(description='magnetic decision attack for model inversion')
    parser.add_argument('--target_model', default='FaceNet64', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--target_model_path', default='models/target_ckp/FaceNet64_88.50.tar', type=str, 
                        help='models/target_ckp/FaceNet64_88.50.tar | models/facescrub/target_model/facescrub_FaceNet64_93.12.tar')
    parser.add_argument('--evaluator_model', default='FaceNet', help='VGG16 | IR152 | FaceNet64| FaceNet')
    parser.add_argument('--evaluator_model_path', default='models/target_ckp/FaceNet_95.88.tar', type=str,
                        help=' models/target_ckp/FaceNet_95.88.tar | models/facescrub/target_model/facescrub_FaceNet_97.37(all).tar')
    parser.add_argument('--generator_model_path', default='models/improved_celeba_G.tar', type=str, 
                        help='models/improved_celeba_G.tar | models/ffhq_improved_G_targetfacescrub_0.75.tar')
    parser.add_argument('--device', type=str, default='0', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--experiment_name', type=str, default='default_test1', help='experiment name for experiment directory')
    parser.add_argument('--config_file', default='config.yaml', type=str, help='config file that has attack params')
    parser.add_argument('--private_imgs_path', default='attack_imgs/', type=str, help='Path to private images')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--n_classes', default=1000, type=int, help='num of classes of target model') # celeba 1000 facescrub 200
    parser.add_argument('--n_classes_evaluator', default=1000, type=int, help='num of classes of target model')  # celeba 1000 facescrub 530
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    logger = get_logger()
    logger.info(args)
    logger.info("=> loading models ...")
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # loading attack params
    with open (args.config_file) as config_file:
        attack_params = yaml.safe_load(config_file)
    print(attack_params)    
    #loading the models
    n_classes = args.n_classes

    if args.target_model.startswith("VGG16"):
        target_model = VGG16(n_classes)
    elif args.target_model.startswith('IR152'):
        target_model = IR152(n_classes)
    elif args.target_model == "FaceNet64":
        target_model = FaceNet64(n_classes)
        
    path_target_model = args.target_model_path
    target_model = target_model.cuda()
    target_model = torch.nn.DataParallel(target_model).cuda()
    ckp_target_model = torch.load(path_target_model)

    target_model.load_state_dict(ckp_target_model['state_dict'], strict=False)
    
    path_G = args.generator_model_path
    G = Generator(attack_params['z_dim'])
    G = torch.nn.DataParallel(G).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)

    if args.evaluator_model == 'FaceNet':
        E = FaceNet(args.n_classes_evaluator)
    elif args.evaluator_model == 'FaceNet64':
        E = FaceNet64(args.n_classes_evaluator)
    
    E = E.cuda()
    #E = torch.nn.DataParallel(E).cuda()
    path_E = args.evaluator_model_path
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    
    target_model.eval()
    G.eval()
    E.eval()

    # prepare working dirs
    attack_imgs_dir = 'decision/attack_imgs/'+args.experiment_name
    os.makedirs(attack_imgs_dir, exist_ok = True)

    eval_model = E
    # do the attack
    magnetic_attack(attack_params,
                    target_model,
                    eval_model, # E
                    G,
                    attack_imgs_dir,
                    args.private_imgs_path
                    )
