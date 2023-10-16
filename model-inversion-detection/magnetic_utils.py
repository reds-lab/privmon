from utils import *
import utils
from classify import *
import torch
import time
import random
import os, logging
import numpy as np
from os import listdir
from os.path import isdir, join

# Import Feed
import sys
sys.path.append(os.path.join(sys.path[0], "../../"))
from monitor.generator import Feed
kafka_addr = "localhost:9092"

feed = Feed(kafka_addr)

def decision_Evaluator(imgs,model,score=False, target=None, criterion = None):
    with torch.no_grad():
        T_out = model(utils.low2high(imgs))[-1]
        #T_out = model(imgs)[-1]
        val_iden = torch.argmax(T_out, dim=1).view(-1)

    if score:
        return val_iden,criterion(T_out, target)
    else:
        return val_iden
    
def decision(imgs,model,score=False, target=None, criterion = None):
    feed.feed2kafka(imgs)
    with torch.no_grad():
        T_out = model(imgs)[-1]
        val_iden = torch.argmax(T_out, dim=1).view(-1)

    if score:
        return val_iden,criterion(T_out, target)
    else:
        return val_iden 

def is_target_class(idens, target, model,score=False, criterion = None):
    if score:
        target_class_tensor = torch.tensor([target]).cuda()
        val_iden, score = decision(idens,model, score, target_class_tensor, criterion = criterion )
    else:
        val_iden = decision(idens,model)
    val_iden[val_iden != target] = -1
    val_iden[val_iden == target] = 1
    return val_iden

#generate the first random intial points for N different labels
def gen_initial_points_untargeted(num_idens, batch_size, G, model, min_clip, max_clip, z_dim):
    print('Generating initial points for attacked target classes: Untargeted Attack')
    initial_points = {}#{194:1}
    max_idens_reached = False
    current_iter = 0
    with torch.no_grad():
        while True:
            z = torch.randn(batch_size, z_dim).cuda().float().clamp(min=min_clip, max=max_clip)
            first_img = G(z)
            # our target class is the now the current class of the generated image
            target_classes = decision(first_img, model)
            
            for i in range(target_classes.shape[0]):
                current_label = target_classes[i].item()
                if current_label in initial_points:
                    continue
                
                initial_points[current_label] = z[i]
            
                if len(initial_points) == num_idens:
                    break
            print("iter {}: current number of distinct labels {}".format(current_iter, len(initial_points)))
            current_iter += 1
            if len(initial_points) == num_idens:
                break
    #initial_points.pop(194, None)
    return initial_points


#generate the initial points for labels from 0 to N
def gen_initial_points_targeted(num_idens, batch_size, G, model, min_clip, max_clip, z_dim, iden_range_min, iden_range_max):
    num_idens = iden_range_max - iden_range_min +1
    print('Generating initial points for attacked target classes: Targeted Attack')
    initial_points = {}
    max_idens_reached = False
    current_iter = 0
    with torch.no_grad():
        while True:
            z = torch.randn(batch_size, z_dim).cuda().float().clamp(min=min_clip, max=max_clip)
            first_img = G(z)
            # our target class is the now the current class of the generated image
            target_classes = decision(first_img, model)
            
            for i in range(target_classes.shape[0]):
                current_label = target_classes[i].item()
                if current_label in initial_points or current_label < iden_range_min or current_label >= iden_range_max:
                    continue
                
                initial_points[current_label] = z[i]
            
                if len(initial_points) == num_idens:
                    break
            print("iter {}: current number of distinct labels {}".format(current_iter, len(initial_points)))
            current_iter += 1
            if len(initial_points) == num_idens:
                break
    return initial_points


# attack the same identities as a given experiment
def gen_idens_as_exp(path_to_exp, batch_size, G, model, min_clip, max_clip, z_dim):
    print("attacking identities from experiment: ", path_to_exp)
    idens = [ int(f.split('_')[1]) for f in listdir(path_to_exp) if isdir(join(path_to_exp, f))]
    num_idens = len(idens)
    print("{} dirs found in the experiment".format(len(idens)))
    
    attacked_idens = {}
    for iden in idens:
        attacked_idens[iden] = 1
    
    initial_points = {}
    current_iter = 0
    with torch.no_grad():
        while True:
            z = torch.randn(batch_size, z_dim).cuda().float().clamp(min=min_clip, max=max_clip)
            first_img = G(z)
            # our target class is the now the current class of the generated image
            target_classes = decision(first_img, model)
            
            for i in range(target_classes.shape[0]):
                current_label = target_classes[i].item()
                if current_label in initial_points or current_label not in  attacked_idens:
                    continue
                
                initial_points[current_label] = z[i]
            
                if len(initial_points) == num_idens:
                    break
            print("iter {}: current number of distinct labels {}".format(current_iter, len(initial_points)))
            current_iter += 1
            if len(initial_points) == num_idens:
                break

    return initial_points

# get the same initial point for each identity in a given experiment
def gen_initial_points_from_exp(path_to_exp):
    print("getting initial points from experiment: ", path_to_exp)
    initial_pont_file = 'initial_z_point.npy'
    idens_dirs = [ f for f in listdir(path_to_exp) if isdir(join(path_to_exp, f))]
    print("{} dirs found in the experiment".format(len(idens_dirs)))
    points = {}
    for iden in idens_dirs:
        npfile = join(path_to_exp, iden,initial_pont_file)
        iden = int(iden.split('_')[1])
        point = np.load(npfile)
        points[iden] = torch.from_numpy(point)

    print("loaded idens: {}".format([k for k in points]))
    return points