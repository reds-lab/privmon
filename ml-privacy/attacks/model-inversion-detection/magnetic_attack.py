from torchvision.utils import save_image
from torch.autograd import grad
import torch
import time
import random
import os, logging
import numpy as np
from magnetic_utils import *
import shutil

def gen_points_on_sphere(current_point, points_count, sphere_radius):

    points_shape = (points_count,) + current_point.shape
    perturbation_direction = torch.randn(*points_shape).cuda()
    dims = tuple([i for i in range(1, len(points_shape))])
    perturbation_direction = (sphere_radius/ torch.sqrt(torch.sum(perturbation_direction ** 2, axis = dims, keepdims = True))) * perturbation_direction
    sphere_points = current_point + perturbation_direction
    return sphere_points, perturbation_direction


def magnetic_attack_single_target(current_point, target_class, current_loss, G,
                    target_model, evaluator_model, attack_params, criterion, current_iden_dir ):
    current_iter = 0
    last_iter_when_radius_changed = 0
    log_file = open(os.path.join(current_iden_dir,'train_log'),'w')
    losses= []
    target_class_tensor = torch.tensor([target_class]).cuda()
    current_sphere_radius = attack_params['current_sphere_radius']
    # Outer loop handle all radii
    
    local_begin_idx = 0

    while  current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
        # inner loop handle one single radius
        while current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
            
            new_radius = False
            step_size = min(current_sphere_radius / 3,3)
            
            # sample points on the sphere
            new_points, perturbation_directions = gen_points_on_sphere(current_point,attack_params['sphere_points_count'], current_sphere_radius)
            
            import pickle

            if current_iter < 10 or (current_iter > 1000 and current_iter < 1010) :
                print('=== current iteration is : ', current_iter)
                
                folder = "./pickle_save/benign/"
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                name = folder + str(current_iter) + ".pt"
                torch.save(G(new_points), name)
                
            
            new_points_classification = is_target_class(G(new_points),target_class,target_model) 
        
            # handle case where all(or some percentage) new points lie in decision boundary. We inc sphere size
            if new_points_classification.sum() > 0.75 * attack_params['sphere_points_count'] :# == attack_params['sphere_points_count']:
                save_tensor_images(G(current_point.unsqueeze(0))[0].detach(),
                                   os.path.join(current_iden_dir,                                 "last_img_of_radius_{:.4f}_iter_{}.png".format(current_sphere_radius, current_iter)))
                # expand the current sphere radius
                current_sphere_radius = current_sphere_radius * attack_params['sphere_expansion_coeff']
                
                log_file.write("new sphere radius at iter: {} ".format(current_iter))
                new_radius = True
                last_iter_when_radius_changed = current_iter
            
            
            # get the magnet direction, which is the mean of direction multiplied by magnetic polarity(decision) over the norm of perturnations
            if attack_params['repulsion_only'] == True:
                new_points_classification_neg_only = (new_points_classification - 1)/2
                grad_direction = torch.mean(new_points_classification_neg_only.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius
            else:
                grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius

            # move the current point with stepsize towards grad_direction
            current_point_new = current_point + step_size * grad_direction
            current_point_new = current_point_new.clamp(min=attack_params['point_clamp_min'], max=attack_params['point_clamp_max'])
            
            current_img = G(current_point_new.unsqueeze(0))
            if is_target_class(current_img,target_class,target_model)[0] == -1:
                log_file.write("current point is outside target class boundary")
                break

            current_point = current_point_new
            _,current_loss = decision(current_img,target_model,score=True, criterion = criterion, target=target_class_tensor)

            if current_iter % 50 == 0 or (current_iter < 200 and current_iter % 20 == 0  ) :
                save_tensor_images(current_img[0].detach(), os.path.join(current_iden_dir, "iter{}.png".format(current_iter)))
            if new_radius:
                point_before_inc_radius = current_point.clone()
                break
            eval_decision = decision_Evaluator(current_img ,evaluator_model)
            correct_on_eval = True if  eval_decision==target_class else False

            iter_str = "iter: {}, current_sphere_radius: {}, step_size: {:.2f} sum decisions: {}, loss: {:.4f}, eval predicted class {}, classified correct on Eval {}".format(
                current_iter,current_sphere_radius, step_size,
                new_points_classification.sum(),
                current_loss.item(),
                eval_decision,
                correct_on_eval)
            
            log_file.write(iter_str+'\n')
            losses.append(current_loss.item())
            current_iter +=1

    log_file.close()
    acc = 1 if decision_Evaluator(G(current_point.unsqueeze(0)),evaluator_model)==target_class  else 0
    return acc


def magnetic_attack(attack_params,
                    target_model,
                    evaluator_model,
                    generator_model,
                    attack_imgs_dir,
                    private_domain_imgs_path):
    
    
    if 'targets_from_exp' in attack_params:
        points = gen_initial_points_from_exp(attack_params['targets_from_exp'])
    elif 'gen_idens_as_exp' in attack_params:
        points = gen_idens_as_exp(attack_params['gen_idens_as_exp'],
                                           attack_params['batch_dim_for_initial_points'],
                                           generator_model,
                                           target_model,
                                           attack_params['point_clamp_min'],
                                           attack_params['point_clamp_max'],
                                           attack_params['z_dim'])
    elif attack_params['targeted_attack']:
        points = gen_initial_points_targeted(attack_params['num_targets'],
                                           attack_params['batch_dim_for_initial_points'],
                                           generator_model,
                                           target_model,
                                           attack_params['point_clamp_min'],
                                           attack_params['point_clamp_max'],
                                           attack_params['z_dim'],
                                           attack_params['iden_range_min'],
                                           attack_params['iden_range_max'])
    else:
        points = gen_initial_points_untargeted(attack_params['num_targets'],
                                           attack_params['batch_dim_for_initial_points'],
                                           generator_model,
                                           target_model,
                                           attack_params['point_clamp_min'],
                                           attack_params['point_clamp_max'],
                                           attack_params['z_dim'])
    
    #points.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    correct_on_eval = 0
    current_iter = 0
    
    folder = "./pickle_save/"
    if os.path.isdir(folder):
        print("Exists")
    else:
        print("Doesn't exists")
        os.mkdir(folder)
        
    for idx, target_class in enumerate(points):
        current_iter += 1
        current_point = points[target_class].cuda()
        print(" {}/{}: attacking iden {}".format(current_iter, len(points), target_class))
        target_class_tensor = torch.tensor([target_class]).cuda()

        # save the first generated image, and current point (z) to the iden_dir
        current_iden_dir = os.path.join(attack_imgs_dir,"iden_{}".format(target_class))
        os.makedirs(current_iden_dir, exist_ok=True)
        first_img = generator_model(current_point.unsqueeze(0))
        save_tensor_images(first_img[0].detach(), os.path.join(current_iden_dir, "original_first_point.png".format(current_iter)))
        np.save(os.path.join(current_iden_dir, 'initial_z_point'),
                current_point.cpu().detach().numpy())
        
        assert is_target_class(first_img,target_class,target_model).item() == 1
        
        _, initial_loss = decision(generator_model(current_point.unsqueeze(0)),target_model,score=True, criterion= criterion, target=target_class_tensor)
        
        correct_on_eval += magnetic_attack_single_target(current_point, target_class, initial_loss, generator_model, target_model, evaluator_model, attack_params, criterion, current_iden_dir )
        current_acc_on_eval = correct_on_eval / current_iter
        print("current acc on eval model: {:.2f}%".format(current_acc_on_eval*100))
        
        if idx ==0 :
            break
    total_acc_on_eval = correct_on_eval / len(points)
    print("total acc on eval model: {:.2f}%".format(total_acc_on_eval*100))
