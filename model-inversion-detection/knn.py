import numpy as np
import matplotlib.pyplot as plt 
import time
import random
import torch
import lpips
from statsmodels.tools.eval_measures import mse

loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization                

def compute_mse(test_sample, queries):
    a = torch.square(test_sample-queries)
    b = a.reshape(a.shape[0], -1)
    c = b.mean(dim=0)
    return c


# Take a vector x and returns the indices of its K nearest neighbors in the training set: train_data
def find_KNN_MSE(test_sample, queries, K): 
    queries_cat = torch.cat(queries, dim=0)
    #l2_dist = compute_mse(test_sample, queries_cat)
    l2_dist = mse(test_sample, queries_cat)
    neighbors = torch.sort(torch.tensor(l2_dist))[0][:K]
    avg_l2_dist = torch.mean(neighbors)
    return avg_l2_dist.numpy()
    

# Take a vector x and returns the indices of its K nearest neighbors in the training set: train_data
def find_KNN_PERC(test_sample, queries, K):
    global loss_fn_vgg
    queries_cat = torch.cat(queries, dim=0)
    percep_loss = loss_fn_vgg(test_sample, queries_cat)
    percep_loss_all = torch.squeeze(percep_loss)
    neighbors = torch.sort(percep_loss_all)[0][:K]
    avg_percept_dist = torch.mean(neighbors)
    return avg_percept_dist.detach().cpu().numpy()


def compute_similarity(test_sample, queries):
    global loss_fn_vgg
    queries_cat = torch.cat(queries, dim=0)
    
    percep_loss = loss_fn_vgg(test_sample, queries_cat)
    percep_loss_all = percep_loss.tolist()
    
    mse_loss = compute_mse(test_sample, queries_cat)
    mse_loss_all = mse_loss.tolist()        
    
    score_p = np.average(np.array(percep_loss_all))
    score_i = np.average(np.array(mse_loss_all))
    return score_p, score_i 

