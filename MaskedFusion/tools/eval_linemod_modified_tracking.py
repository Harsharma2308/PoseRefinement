import _init_paths
import argparse
import os
import random
import icp
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()
import open3d as o3d
import cv2

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

testdataset = PoseDataset_linemod('test', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=8)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
success_count_cpy = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs_ICP_DEL2.txt'.format(output_result_dir), 'w')

import time
start_time = time.time()

t_dis = []
t_itr = []
istracked = False
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx, mask, image = data

    if i in [82, 83, 85,86, 96]:
        pass
        #TODO: Update Tracked status
    else:
        continue
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue

    points, choose, img, target, model_points, idx, mask, image = Variable(points).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(target).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(idx).cuda(), \
                                                                Variable(mask).cuda(), \
                                                                Variable(image)

    inner_time = time.time()
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx, mask)

    '''
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    '''
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)
    #SACHIT TRY LEFT MULTIPLY
    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
    my_mat = quaternion_matrix(my_r)
    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
    my_mat[0:3, 3] = my_t
    my_r_unrefined = quaternion_matrix(my_r)[:3, :3]
    my_t_unrefined = my_t + 0

    if istracked:
        my_r = my_r_last
        my_t = my_t_last
        my_r_unrefined = quaternion_matrix(my_r)[:3, :3]
        my_t_unrefined = my_t + 0

    ICP = False
    if ICP:
        #Refinement
        new_cloud = torch.bmm((points - T), R).contiguous()
        # dummy = points.squeeze().detach().cpu().numpy()
        init_cloud = new_cloud.squeeze().detach().cpu().numpy().copy()
        original_cloud = model_points[0].cpu().detach().numpy().copy()
        # my_r = quaternion_matrix(my_r)[:3, :3]
        # original_cloud_shifted = np.dot(original_cloud, my_r.T) + my_t
        # from IPython import embed; embed()
        delta_T, distances, iterations = icp.icp(original_cloud, init_cloud, max_iterations=20000, tolerance=0.000001)
        t_itr.append(iterations)
        # pcd_src = o3d.geometry.PointCloud()
        # pcd_target = o3d.geometry.PointCloud()
        # pcd_src.points = o3d.utility.Vector3dVector(init_cloud)
        # pcd_target.points = o3d.utility.Vector3dVector(original_cloud)
        # t_itr.append(0)
        # reg_p2p = o3d.pipelines.registration.registration_icp(pcd_target, pcd_src, 0.2, np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 20000, relative_rmse = 1.0e-10, relative_fitness=1.000000e-10))
        # delta_T = reg_p2p.transformation
    
    
        my_mat_final = np.dot(my_mat,delta_T)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)

        my_r = my_r_final
        my_t = my_t_final
    else:
        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
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

    my_r_last = my_r + 0
    my_t_last = my_t + 0
        
    

    post_refineement_ICP = False
    isfail = False
    if post_refineement_ICP:
        
        # my_r_unrefined = quaternion_matrix(my_r)[:3, :3]
        # my_t_unrefined = my_t + 0

        mp = model_points[0].cpu().detach().numpy().copy()
        my_r_sec = quaternion_matrix(my_r)[:3, :3]
        # pred_unrefined = np.dot(model_points, my_r.T) + my_t
        pred_sec = np.dot(mp, my_r_sec.T) + my_t
        target_sec = target[0].cpu().detach().numpy().copy()
        
        if idx[0].item() in sym_list:
            pred_sec = torch.from_numpy(pred_sec.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target_sec = torch.from_numpy(target_sec.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds_sec = knn(target_sec.unsqueeze(0), pred_sec.unsqueeze(0))
            target_sec = torch.index_select(target_sec, 1, inds_sec.view(-1) - 1)
            dis_sec = torch.mean(torch.norm((pred_sec.transpose(1, 0) - target_sec.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis_sec = np.mean(np.linalg.norm(pred_sec - target_sec, axis=1))


        if dis_sec >=  diameter[idx[0].item()]:
            isfail = True
            print("____________\n_____________",i)
            # #Pose Refinement Refinement
            new_cloud = torch.bmm((points - T), R).contiguous()
            # dummy = points.squeeze().detach().cpu().numpy()
            init_cloud = new_cloud.squeeze().detach().cpu().numpy()
            original_cloud = model_points[0].cpu().detach().numpy().copy()
            # my_r = quaternion_matrix(my_r)[:3, :3]
            # original_cloud_shifted = np.dot(original_cloud, my_r.T) + my_t

            delta_T, distances, iterations = icp.icp(original_cloud, init_cloud, max_iterations=50000, tolerance=0.0000001)
            
            # pcd_src = o3d.geometry.PointCloud()
            # pcd_target = o3d.geometry.PointCloud()
            # pcd_src.points = o3d.utility.Vector3dVector(init_cloud)
            # pcd_target.points = o3d.utility.Vector3dVector(original_cloud)

            # reg_p2p = o3d.pipelines.registration.registration_icp(pcd_target, pcd_src, 0.01, np.eye( 4))#,o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 20000, relative_rmse = 1.0e-10, relative_fitness=1.000000e-10))
            # delta_T = reg_p2p.transformation
            my_mat_final = np.dot(my_mat,delta_T)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)

            my_r = my_r_final
            my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    # pred_unrefined = np.dot(model_points, my_r.T) + my_t
    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()


    #Image Printing
    img = image.squeeze().detach().cpu().numpy().copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cam_cx = 325.26110
    cam_cy = 242.04899
    cam_fx = 572.41140
    cam_fy = 573.57043
    cam_mat = np.eye(3)
    cam_mat[0,0] = cam_fx
    cam_mat[1,1] = cam_fy
    cam_mat[0,2] = cam_cx
    cam_mat[1,2] = cam_cy
    # from IPython import embed; embed()

    p_pred_unrefined, _ = cv2.projectPoints(model_points, cv2.Rodrigues(my_r_unrefined)[0], my_t_unrefined, cam_mat, None)
    p_pred, _ = cv2.projectPoints(model_points, cv2.Rodrigues(my_r)[0], my_t, cam_mat, None)
    p_tar, _ = cv2.projectPoints(target, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), cam_mat, None)

    save_imgs = True
    if save_imgs:
        if i<100:
            for j in range(500):
                x = p_pred[j,0,0]
                y = p_pred[j,0,1]
                x1 = p_tar[j,0,0]
                y1 = p_tar[j,0,1]
                x2 = p_pred_unrefined[j,0,0]
                y2 = p_pred_unrefined[j,0,1]
                if x<0 or x>img.shape[1] or y<0 or y>img.shape[0]:
                    continue
                img = cv2.circle(img, (x,y), radius=0, color=(0, 255, 255), thickness=-1)
                if x1<0 or x1>img.shape[1] or y1<0 or y1>img.shape[0]:
                    continue
                img = cv2.circle(img, (x1,y1), radius=0, color=(255, 255, 0), thickness=-1)

                if x2<0 or x2>img.shape[1] or y2<0 or y2>img.shape[0]:
                    continue
                col = (255, 0, 0)
                if istracked:
                    col = (0,255,0)
                img = cv2.circle(img, (x2,y2), radius=0, color=col, thickness=-1)

            cv2.imwrite("images_overlapped/Tracked_"+str(i)+".png", img)

    

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    t_dis.append(dis)
    if dis < 0.05:#diameter[idx[0].item()]:
        istracked = True
        success_count[idx[0].item()] += 1
        success_count_cpy[idx[0].item()] += 1
        
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))

        if isfail and post_refineement_ICP:
            success_count[idx[0].item()] -= 1
            for j in range(500):
                x = p_pred[j,0,0]
                y = p_pred[j,0,1]
                x1 = p_tar[j,0,0]
                y1 = p_tar[j,0,1]
                x2 = p_pred_unrefined[j,0,0]
                y2 = p_pred_unrefined[j,0,1]
                if x<0 or x>img.shape[1] or y<0 or y>img.shape[0]:
                    continue
                img = cv2.circle(img, (x,y), radius=0, color=(0, 255, 255), thickness=-1)
                if x1<0 or x1>img.shape[1] or y1<0 or y1>img.shape[0]:
                    continue
                img = cv2.circle(img, (x1,y1), radius=0, color=(255, 255, 0), thickness=-1)

                if x2<0 or x2>img.shape[1] or y2<0 or y2>img.shape[0]:
                    continue
                img = cv2.circle(img, (x2,y2), radius=0, color=(255, 0, 0), thickness=-1)

            cv2.imwrite("images_overlapped/NEWSUCESS_"+str(i)+".png", img)
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

print("### %s seconds ###" % (time.time() - start_time))

t_dis = np.array(t_dis)
print("Average dis = ", np.mean(t_dis))
if ICP and len(t_itr):
    t_itr = np.array(t_itr)
    print("Average Iterations = ", np.mean(t_itr))
    


for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
# print('ALL success rate POST ICP: {0}'.format(float(sum(success_count_cpy)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.write('Average dis: {0}\n'.format(np.mean(t_dis)))
fw.close()
#TIME Difference 32.8sec vs 52 seconds for 1335 images
#ICP Fails -> 19
#Post refine avg dis 0.005437343089536802
#MF avg dis 0.005505017409778832
# Just ICP 0,055
#No refine dist  0.07  ;0.06446128533301987