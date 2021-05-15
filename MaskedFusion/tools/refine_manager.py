import torch
import numpy as np
from time import time
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from torch.autograd import Variable
import copy
class RefineManager():

    def __init__(self,refine_method):
        if(refine_method=='posenet'):
            self.refine = refine_posenet
        elif(refine_method=='icp'):
            self.refine = refine_icp

    def refine_icp(self,refine_args):
        pass
    
    def refine_posenet(self,refine_args):
        iteration = refine_args.iteration
        my_t,my_r = refine_args.t,refine_args.r
        num_points = refine_args.num_points
        cloud = refine_args.cloud
        refiner = refine_args.refiner_network
        emb = refine_args.emb
        index = refine_args.index


        for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                time_refiner = time.time()
                pred_r, pred_t = refiner(new_cloud, emb, index)
                print("--- RE %s seconds ---" % (time.time() - time_refiner))
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final
        

        