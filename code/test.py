'''
The code is modified from the implementation of Zooming Slow-Mo:
https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020
'''
import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.data_sampler import DistIterSampler

# import configs.configs as option
from utils import util
from configs import option

from dataset import create_dataloader, create_dataset
from models import create_model
from pdb import set_trace as bp

from utils.util import calculate_psnr, calculate_ssim
from utils.utils_y import calc_psnr_pytorch, calc_ssim_pytorch, calc_psnr_pytorch_float, calc_ssim_pytorch_float, calc_lpips_pytorch_float, MetricTracker,get_color_map
def g(*args, **kwargs):  # debug no-op (dmfq removed for release)
    pass

from os.path import join,dirname,basename
import cv2
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
# import torchshow as ts


import numpy as np
import tqdm
import cv2
import numpy as np
from typing import List, Dict, Tuple

import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()

import flow_vis
import multiprocessing as mp
from einops import rearrange

# 设置多进程的启动方式为 'spawn'
mp.set_start_method('spawn', force=True)



'''
Usage:
    python test.py --yml_file configs/test_adobe.yml --save_vis
    python test.py --yml_file configs/test_gopro.yml --save_vis
'''

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


from typing import List


def hw3_to_nh_nw_3(arr,spatial_upscale=None):
    # 假设原始数组为arr，h和w分别为原始数组的高度和宽度
    h, w, _ = arr.shape
    # 扩展形状为（4h，4w，3）的数组
    expanded_arr = np.repeat(arr, spatial_upscale, axis=0)
    expanded_arr = np.repeat(expanded_arr, spatial_upscale, axis=1)
    # 重塑形状为（4h，4w，3）的数组
    final_arr = np.reshape(expanded_arr, (spatial_upscale*h, spatial_upscale*w, 3))
    return final_arr



def convert_flow_to_vis_hw2(flo):
    '''
    flo :   ([1, 2, 256, 256])
    vgrid:  ([1, 2, 256, 256])
    '''
    B, _, H, W = flo.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat((xx, yy), 1).float()

    if flo.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0 # ([1, 2, 256, 256])
    # vgrid = vgrid.clamp(-1, 1)
    return vgrid


def post_process_in_test(out_dict,current_step,opt,testset_opt,test_iter_id,metric_dict,real255_dict,fake255_dict,erro255_dict, lq_dict,logger,pre_sce_name,save_vis,verbose,metrics):

    cal_lpips    = testset_opt.get('cal_lpips', False)  # 字典如果存在键，返回键值，如果不存在，返回 False
    vis_flow    = testset_opt.get('vis_flow', False)  # 字典如果存在键，返回键值，如果不存在，返回 False
    vis_ev      = testset_opt.get('vis_ev', False)  # 字典如果存在键，返回键值，如果不存在，返回 False

    lq = out_dict['LQ']             # ([bs, 2, 3, h, w])
    lq_ev_0t_l = out_dict['LQs_ev_0t_l']    # List:len=3     1,1,16,256,256   1,1,16,256,256  ...
    lq_ev_t1_l = out_dict['LQs_ev_t1_l']

    real_H = out_dict['GT']         # List:len=9  bs,1,3,hh,ww
    fake_H = out_dict['Fake']       # List:len=9  bs,3,hh,ww   
    sce_name_l = out_dict['sce_name_l']         # List[List[str]:len=bs]:len=9 [['GOPR9653'], ['GOPR9653'],...]]
    gt_png_name_l = out_dict['gt_png_name_l']   # List[List[str]:len=bs]:len=9 [['0.png'], ['1.png'],...]]
    try:
        lq_png_name_l = out_dict['lq_png_name_l']
    except:
        pass
    flow_t0_l = out_dict['preds_flow_t0_l'] # List:len=3     1,2,256,256     1,2,256,256     ...
    flow_t1_l = out_dict['preds_flow_t1_l'] # List:len=3     1,2,256,256     1,2,256,256     ...

    # flow_t0_l = [convert_flow_to_vis_hw2(flo)  for flo in flow_t0_l] # 因为原来的值 range 不是vis用的 range,   List:len=3     1,2,256,256     1,2,256,256     ...
    # flow_t1_l = [convert_flow_to_vis_hw2(flo)  for flo in flow_t1_l] # List:len=3     1,2,256,256     1,2,256,256     ...
    # real_H = model.real_H_l # List  bs,1,3,hh,ww
    # fake_H = model.fake_H   # List  bs,3,hh,ww
    # sce_name_l    = model.sce_name_l        # List[List:len=bs]:len=9
    # gt_png_name_l = model.gt_png_name_l     # List[List:len=bs]:len=9

    lq = lq                                                           #    bs,2,c, h, w  
    real_H = torch.cat([itm[:,:,:,:] for itm in real_H], dim=1)       # -> bs,9,c,hh,ww
    fake_H = torch.cat([itm[:,None,:,:,:] for itm in fake_H], dim=1)  # -> bs,9,c,hw,ww

    # print(f"==>> real_H.shape: {real_H.shape}")
    # print(f"==>> fake_H.shape: {fake_H.shape}")
    assert real_H.shape == fake_H.shape, f"real_H.shape: {real_H.shape} != fake_H.shape: {fake_H.shape}"
    bs,ns,c,hh,ww=real_H.shape
    # print(f"==>> real_H.shape: {real_H.shape}")
    middle_idx = ns//2  # 3: 0,1,2  middle_idx=1,    5: 0,1,2,3,4 middle_idx=2,   9: 0,1,2,3,4,5,6,7,8 middle_idx=4 

    h_st = opt['datasets']['test']['h_st']  # 0    # 20
    h_ed = opt['datasets']['test']['h_ed']  # 440  # 420
    w_st = opt['datasets']['test']['w_st']  # 0    # 20
    w_ed = opt['datasets']['test']['w_ed']  # 680  # 620

    # ssim_dict = {}      #  ssim_dict = Dict[str,List[Tensor]]
    for b in range(bs):
        for n in range(ns):
            sce_name = sce_name_l[n][b]         # str
            gt_pnt_name = gt_png_name_l[n][b]   # str

            
            # psnr_flt = calc_psnr_pytorch((real_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]*255).to(torch.uint8), # .to(torch.uint8)
            #                              (fake_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]*255).to(torch.uint8))  # float
            
            # ssim_flt = calc_ssim_pytorch((real_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]*255).to(torch.uint8),
            #                              (fake_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]*255).to(torch.uint8))  # float
            
            psnr_flt = calc_psnr_pytorch_float((real_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]), # .to(torch.uint8)  NCHW  RGB
                                            (fake_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]))  # float
            
            ssim_flt = calc_ssim_pytorch_float((real_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]),                     # NCHW  RGB
                                            (fake_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]))  # flaot
            
            if cal_lpips:
                lpips_flt = calc_lpips_pytorch_float((real_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]),                     # NCHW  RGB
                                                (fake_H[b,n,:,h_st:h_ed,w_st:w_ed][None,:,:,:]))  # flaot
            else:
                lpips_flt = 0
            metrics.update('psnr_score_ave',psnr_flt)
            metrics.update('ssim_score_ave',ssim_flt)
            metrics.update('lpips_score_ave',lpips_flt)

            metrics.update(f'{sce_name}/psnr_score_ave',psnr_flt)
            metrics.update(f'{sce_name}/ssim_score_ave',ssim_flt)
            metrics.update(f'{sce_name}/lpips_score_ave',lpips_flt)


            if n==0 or n ==ns:
                metrics.update('psnr_score_ends',psnr_flt)
                metrics.update('ssim_score_ends',ssim_flt)
                metrics.update('lpips_score_ends',lpips_flt)
                
                metrics.update(f'{sce_name}/psnr_score_ends',psnr_flt)
                metrics.update(f'{sce_name}/ssim_score_ends',ssim_flt)
                metrics.update(f'{sce_name}/lpips_score_ends',lpips_flt)

            if n==0 or n ==ns or n == middle_idx:
                metrics.update('psnr_score_ctr',psnr_flt)
                metrics.update('ssim_score_ctr',ssim_flt)
                metrics.update('lpips_score_ctr',lpips_flt)
                metrics.update(f'{sce_name}/psnr_score_ctr',psnr_flt)
                metrics.update(f'{sce_name}/ssim_score_ctr',ssim_flt)
                metrics.update(f'{sce_name}/lpips_score_ctr',lpips_flt)


            if save_vis:
                SAVEVIS=None
                real255 = (real_H[b,n,[2,1,0],h_st:h_ed,w_st:w_ed].permute(1,2,0).clamp(0, 1)*255).to(torch.uint8).detach().cpu().numpy()
                fake255 = (fake_H[b,n,[2,1,0],h_st:h_ed,w_st:w_ed].permute(1,2,0).clamp(0, 1)*255).to(torch.uint8).detach().cpu().numpy()

                if vis_ev:
                    ev_0t = lq_ev_0t_l[n][b][0,:,:,:]  # 16,256,256
                    ev_t1 = lq_ev_t1_l[n][b][0,:,:,:]  # 16,256,256

                    vmin_voxel = -1
                    vmax_voxel = 1
                    ev_0t_hw1 = ev_0t.sum(dim=0)[:,:,None] # 256,256,1
                    ev_t1_hw1 = ev_t1.sum(dim=0)[:,:,None] # 256,256,1
                    ev_0t_255 = ((ev_0t_hw1.clamp(vmin_voxel,vmax_voxel) -  vmin_voxel)/(vmax_voxel-vmin_voxel) *255).to(torch.uint8).detach().cpu().numpy()
                    ev_t1_255 = ((ev_t1_hw1.clamp(vmin_voxel,vmax_voxel) -  vmin_voxel)/(vmax_voxel-vmin_voxel) *255).to(torch.uint8).detach().cpu().numpy()
                    ev_0t_255 = np.repeat(ev_0t_255, 3, axis=2)  # h,w,1 -> h,w,3
                    ev_t1_255 = np.repeat(ev_t1_255, 3, axis=2)  # h,w,1 -> h,w,3

                    ev_voxel_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'ev_voxel',gt_pnt_name)
                    os.makedirs(dirname(ev_voxel_save_path),exist_ok=True)

                if vis_flow:
                # flow:  float32, shape (H, W, 2);    flow_color_hw3_BGR: uint8,   shape (H, W, 3)
                # flow_color_hw3_BGR=flow_vis.flow_to_color(flow, convert_to_bgr=False)[:,:,[2,1,0]]    # h,w,3,  BGR
                    flow255_t0 = flow_t0_l[n][b]  # 2,256,256
                    flow255_t1 = flow_t1_l[n][b]  # 2,256,256
                    flow255_t0 = rearrange(flow255_t0, 'c h w -> h w c').detach().cpu().numpy() # 256,256,2
                    flow255_t1 = rearrange(flow255_t1, 'c h w -> h w c').detach().cpu().numpy() # 256,256,2
                    # g(flow255_t0)
                    # flow:  float32, shape (H, W, 2);    flow_color_hw3_BGR: uint8,   shape (H, W, 3)
                    flow_t0_color_hw3_BGR=flow_vis.flow_to_color(flow255_t0, convert_to_bgr=False)[:,:,[2,1,0]]    # h,w,3,  BGR
                    flow_t1_color_hw3_BGR=flow_vis.flow_to_color(flow255_t1, convert_to_bgr=False)[:,:,[2,1,0]]    # h,w,3,  BGR
                    flow_t0_255 = flow_t0_color_hw3_BGR
                    flow_t1_255 = flow_t1_color_hw3_BGR
                    flow_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'flow',gt_pnt_name)
                    os.makedirs(dirname(flow_save_path),exist_ok=True)

                    ev255 = np.hstack((ev_0t_255,ev_t1_255))
                    flow255 = np.hstack((flow_t0_255,flow_t1_255))
                    # cv2.imwrite(ev_voxel_save_path,np.hstack((ev_0t_255,ev_t1_255)))
                    # cv2.imwrite(flow_save_path,np.hstack((flow_t0_255,flow_t1_255)))
                    cv2.imwrite(flow_save_path,np.vstack( (   ev255  ,    flow255   )  )     )


                real_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'gt',gt_pnt_name)
                fake_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'EvSTVSR',gt_pnt_name)
                
                vim,vmax = 0.0, 0.4
                error_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,f'error_vmin{vim}_vmax{vmax}',gt_pnt_name)


                

                os.makedirs(dirname(real_save_path),exist_ok=True)
                os.makedirs(dirname(fake_save_path),exist_ok=True)
                os.makedirs(dirname(error_save_path),exist_ok=True)



                cv2.imwrite(real_save_path,real255)
                cv2.imwrite(fake_save_path,fake255)



                # 计算误差图（绝对差值）
                error_hw3 = np.abs(real255 / 255.0 - fake255 / 255.0)
                error_hw = np.mean(error_hw3, axis=2)
                erro255 = get_color_map(error_hw,vmin=vim, vmax=vmax)
                cv2.imwrite(error_save_path,erro255)

                
                h_ed,w_ed = opt['datasets']['test']['h_ed'], opt['datasets']['test']['w_ed']     # adobe: 312,600  erf: 440,680
                if n==0:
                    #* LR
                    tmp_gt_pnt_name = gt_pnt_name
                    # try:
                    tmp_gt_pnt_name = out_dict['lq_png_name_l'][n][b]
                    # except:
                    #     pass
                    lq0_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'LR',tmp_gt_pnt_name)
                    os.makedirs(dirname(lq0_save_path),exist_ok=True)
                    lq_255= (lq[b,0,[2,1,0],:,:].permute(1,2,0).clamp(0, 1)*255).to(torch.uint8).detach().cpu().numpy()
                    cv2.imwrite(lq0_save_path,lq_255)
                    #* LR_up
                    lq0_save_path_up = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'LR_up',tmp_gt_pnt_name)
                    os.makedirs(dirname(lq0_save_path_up),exist_ok=True)
                    lq_255_up = hw3_to_nh_nw_3(lq_255,spatial_upscale=opt['datasets']['test']['spatial_upscale'])[:h_ed,:w_ed,:]  # todo bug
                    cv2.imwrite(lq0_save_path_up,lq_255_up)

                # elif n==ns-1:
                #     #* LR
                #     lq1_save_path = join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',sce_name,'LR',gt_pnt_name)
                #     lq_255 = (lq[b,1,[2,1,0],:,:].permute(1,2,0).clamp(0, 1)*255).to(torch.uint8).detach().cpu().numpy()
                #     lq_255_up = hw3_to_nh_nw_3(lq_255,spatial_upscale=opt['datasets']['test']['spatial_upscale'])[:h_ed,:w_ed,:]
                #     cv2.imwrite(lq1_save_path,lq_255_up)

                # save sce psnr
                # if sce_name not in metric_dict.keys():
                if n <= ns-1:
                    if sce_name not in metric_dict:
                        metric_dict.update({sce_name:{'psnr':[psnr_flt],'ssim':[ssim_flt]}})
                        real255_dict.update({sce_name: [real255]})
                        fake255_dict.update({sce_name: [fake255]})
                        erro255_dict.update({sce_name: [erro255]})
                        lq_dict.update({sce_name: [lq_255_up]})
                    else:
                        metric_dict[sce_name]['psnr'].append(psnr_flt)
                        metric_dict[sce_name]['ssim'].append(ssim_flt)

                        real255_dict[sce_name].append(real255)
                        fake255_dict[sce_name].append(fake255)
                        erro255_dict[sce_name].append(erro255)
                        lq_dict[sce_name].append(lq_255_up)

            # ssim_array[test_iter_id,b,n] = 0
    if save_vis:
        if test_iter_id != 0:
            if sce_name != pre_sce_name:
                #! logger
                # logger.info('\n'+
                #     f'sce_name: {pre_sce_name}'+
                #     f'  psnr_score_ctr: {metrics.avg(f"{pre_sce_name}/psnr_score_ctr"):.2f}    ssim_score_ctr: {metrics.avg(f"{pre_sce_name}/ssim_score_ctr"):.4f}'+
                #     f'  psnr_score_ave: {metrics.avg(f"{pre_sce_name}/psnr_score_ave"):.2f}    ssim_score_ave: {metrics.avg(f"{pre_sce_name}/ssim_score_ave"):.4f}'+
                #     f'  psnr_score_ends: {metrics.avg(f"{pre_sce_name}/psnr_score_ends"):.2f}   ssim_score_ends: {metrics.avg(f"{pre_sce_name}/ssim_score_ends"):.4f}')
                
                logger.info('psnr_score_ctr  ssim_score_ctr  lpips_score_ctr  psnr_score_ave  ssim_score_ave  lpips_score_ave  psnr_score_ends  ssim_score_ends  lpips_score_ends')
                logger.info(f'sce_name: {pre_sce_name:<30}    '+
                            f'{metrics.avg(f"{pre_sce_name}/psnr_score_ctr"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ctr"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ctr"):.4f}    '+
                            f'{metrics.avg(f"{pre_sce_name}/psnr_score_ave"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ave"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ave"):.4f}  '+
                            f'{metrics.avg(f"{pre_sce_name}/psnr_score_ends"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ends"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ends"):.4f}')

                #! save to video mp4
                print(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',pre_sce_name,'gather.mp4')
                gather = [np.hstack([lq_dict[pre_sce_name][i],
                                    fake255_dict[pre_sce_name][i],
                                    real255_dict[pre_sce_name][i],
                                    erro255_dict[pre_sce_name][i]
                                    ],
                                    ) for i in range(len(real255_dict[pre_sce_name]))]
                # imgHW3_list_to_mp4_vidgear(join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',pre_sce_name,'LR.mp4'),lq_dict[pre_sce_name],fps= 2)
                # imgHW3_list_to_mp4_vidgear(join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',pre_sce_name,'gt.mp4'),real255_dict[pre_sce_name],fps= 8) # todo
                # imgHW3_list_to_mp4_vidgear(join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',pre_sce_name,'VideoINR.mp4'),fake255_dict[pre_sce_name],fps= 8) # todo
                
                # imgHW3_list_to_mp4_vidgear(join(opt['path']['log'],f'test_current_step{current_step}','test_mini_x4',f'{pre_sce_name}_gather.mp4'),gather,fps= 8) # todo
    

    
    pre_sce_name = sce_name
    if verbose:
        a1 = metrics.avg(f"{sce_name}/psnr_score_ave")
        a2 = metrics.avg(f"{sce_name}/ssim_score_ave")
        a3 = metrics.avg(f"{sce_name}/lpips_score_ave")
        logger.info(f'''iter = {test_iter_id}    sce_name={sce_name}    psnr = {a1:.2f}    ssim = {a2:.4f}    lpips = {a3:.4f}    ns = {ns}''')
    
    # count+=1
    # if count >2:
    #     break
    return metric_dict,real255_dict,fake255_dict,erro255_dict, lq_dict,pre_sce_name,metrics


def test_func(model,current_step,test_set,test_sampler,test_loader, opt,testset_opt,logger,rank,save_vis=True,verbose=True):

    # todo
    metrics = MetricTracker('psnr_score_ctr','ssim_score_ctr','lpips_score_ctr','psnr_score_ave','ssim_score_ave','lpips_score_ave','psnr_score_ends','ssim_score_ends','lpips_score_ends')


    test_size = int(math.ceil(len(test_set) / testset_opt['batch_size']))
    logger.debug(f"==>> test_size: {test_size}")
    test_total_iters = test_size # int(opt['test']['niter'])

    if rank <= 0:
        logger.info(f'Number of test images: {len(test_set):d}, test_total_iters: {test_total_iters:d}')
        # else:
        #     raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    current_test_step = 0
    # for epoch in range(start_epoch, total_epochs + 1):
    for epoch in range(1):


        testdata_shuffle=False
        if testdata_shuffle:
            if opt['dist']:
                test_sampler.set_epoch(epoch)
        else:
            pass


        bs = opt['datasets']['test']['batch_size'] #todo 2
        ns = opt['datasets']['test']['N_frames']+2   # 9 #todo 9

        FORLOOP=None
        #* total=17,619 bs=2     # bs=8  total =4405
        count = 0
        
        metric_dict = {}      #  metric_dict = Dict[Dict[str,List[Tensor]]]    # metric_dict['0000']['psnr']
        real255_dict={}
        fake255_dict={}
        erro255_dict={}
        lq_dict={}
        pre_sce_name=None
        # pre_sce_name='acquarium_08'
        for test_iter_id, test_data in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
            current_test_step += 1
            # if current_test_step > test_total_iters:
            #     break
            # if current_test_step > 4: # for debug
            #     break
            # logger.info(f"==>> test_data['sce_name_l']: {test_data['sce_name_l']}") 
            # logger.info(f"==>> test_data['gt_png_name_l']: {test_data['gt_png_name_l']}")
            #### testing
            # g(test_data)
            model.feed_data(test_data)
            model.test()
            out_dict = model.get_current_visuals()
            
            metric_dict,real255_dict,fake255_dict,erro255_dict,lq_dict,pre_sce_name,metrics = post_process_in_test(out_dict,current_step, opt,testset_opt,test_iter_id,
                                                            metric_dict,real255_dict,fake255_dict,erro255_dict,lq_dict,
                                                            logger,pre_sce_name,save_vis=save_vis,verbose=verbose,metrics=metrics)


    logger.info('psnr_score_ctr  ssim_score_ctr  lpips_score_ctr  psnr_score_ave  ssim_score_ave  lpips_score_ave  psnr_score_ends  ssim_score_ends  lpips_score_ends')
    for pre_sce_name in test_set.video_list: # pre_sce_name只是为了跟两一个代码一样
    #     # metrics.avg(f'{sce_name}/psnr_score_ctr')
        try:
            logger.info(f'sce_name: {pre_sce_name:<30}    '+
                        f'{metrics.avg(f"{pre_sce_name}/psnr_score_ctr"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ctr"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ctr"):.4f}  '+
                        f'{metrics.avg(f"{pre_sce_name}/psnr_score_ave"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ave"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ave"):.4f}  '+
                        f'{metrics.avg(f"{pre_sce_name}/psnr_score_ends"):.2f}  {metrics.avg(f"{pre_sce_name}/ssim_score_ends"):.4f}  {metrics.avg(f"{pre_sce_name}/lpips_score_ends"):.4f}')
        except:
            pass

    test_metirc_dict = {'psnr_score_ctr':metrics.avg('psnr_score_ctr'),
                        'ssim_score_ctr':metrics.avg('ssim_score_ctr'),
                        'lpips_score_ctr':metrics.avg('lpips_score_ctr'),
                        'psnr_score_ave':metrics.avg('psnr_score_ave'),
                        'ssim_score_ave':metrics.avg('ssim_score_ave'), 
                        'lpips_score_ave':metrics.avg('lpips_score_ave'), 
                        'psnr_score_ends':metrics.avg('psnr_score_ends'), 
                        'ssim_score_ends':metrics.avg('ssim_score_ends'),
                        'lpips_score_ends':metrics.avg('lpips_score_ends') }
    return test_metirc_dict


def main():
    #### configs
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='A list of integers.')
    parser.add_argument('--iter', type=int, default=0, help='label only: appears in the output folder name')

    parser.add_argument('--yml_file', type=str, default='configs/test_adobe.yml')
    parser.add_argument('--checkpoint', type=str, default='./saved_checkpoints/evstvsr.pth',
                        help='path to the .pth weights (the same Adobe-trained model is used for both Adobe and GoPro)')
    parser.add_argument('--sgl_sce', type=str)  # test a single scene by name
    parser.add_argument('--exp_name', type=str, default='evstvsr', help='name of the output results subfolder')

    # parser.add_argument('--save_vis', type=bool, default=True) #* add   False True 
    parser.add_argument('--save_vis', action='store_true')
    parser.set_defaults(debug=False) 

    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False) #* add

    parser.add_argument('--launcher', choices=['none', 'pytorch'], 
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # opt = option.parse(args.opt, is_train=False)  #True

    PARSER=None
    yml_file = args.yml_file # 'test_erf_mini.yml' 
    with open(yml_file, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    opt['gpu_ids'] = args.gpu_ids
    opt['name']= args.exp_name

    opt = option.parse(opt, is_train=False)

    #! checkpoint weights (the same Adobe-trained model is used for both Adobe and GoPro)
    opt['path']['pretrain_model_G'] = args.checkpoint

    
    #*! ddp
    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #! resume
    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, 
                                  loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume configs
    else:
        resume_state = None #!


    LOGGER=None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None and opt['is_train']:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() 
                         if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))
        else:
            util.mkdir_and_rename(
                opt['path']['results_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() 
                         if not key == 'results_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        
        if args.debug:
            level = logging.DEBUG
        else:
            level=logging.INFO
        #! setup_logger
        util.setup_logger('inrlog', opt['path']['log'], 'test_' + opt['name'], 
                          level=level,
                          screen=True, tofile=True)    #INFO DEBUG
        logger = logging.getLogger('inrlog')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.
                    format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        # util.setup_logger('inrlog', opt['path']['log'], 'train', 
        #                   level=logging.INFO, screen=True)
        # logger = logging.getLogger('inrlog')
        pass

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    
    #! seed
    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    CREATE_DATASET=None
    #! dataset  dataloader
    #### create train and val dataloader
    dataset_ratio = 1 # 200  # enlarge the size of each epoch


    testset_opt = opt['datasets']['test']
    testset_opt['batch_size'] = testset_opt['batch_size_per_gpu']
    test_set = create_dataset(testset_opt)
    
    


    if opt['dist']:
        test_sampler = DistIterSampler(test_set, world_size, rank, dataset_ratio, shuffle = False) # , shuffle=False
    else:
        test_sampler = None
    test_loader = create_dataloader(test_set, testset_opt, opt, test_sampler)



    assert test_loader is not None

    CREATE_MODEL=None
    #! create_model
    #### create model
    model = create_model(opt)

    # model.print_network()
    #! resume_training
    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_test_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_test_step = 0
        start_epoch = 0

    #! test_pretrained  add!
    if opt['path']['pretrain_model_G'] is not None:
        logger.info('Loding testing pretrained model.')
        model.load()

    #### testing
    logger.info('#### Start testing at epoch: {:d}, iter: {:d}'.format(start_epoch, current_test_step))
    

    current_step=args.iter   # resume_state['iter']
    test_metirc_dict = test_func(model,current_step,test_set,test_sampler,test_loader,   opt,testset_opt,logger,rank,save_vis=args.save_vis, verbose=True)
    

    dc = test_metirc_dict
    
    logger.info(f"#### End testing in testing:  All  {dc['psnr_score_ctr']:.2f},  {dc['ssim_score_ctr']:.4f},  {dc['lpips_score_ctr']:.4f},  {dc['psnr_score_ave']:.2f},  {dc['ssim_score_ave']:.4f},  {dc['lpips_score_ave']:.4f},  {dc['psnr_score_ends']:.2f},  {dc['ssim_score_ends']:.4f},  {dc['lpips_score_ends']:.4f} ")
    # message_test = ''
    # for k, v in dc.items():
    #     message_test += f'{k:s}: {v:.4f} '
    # # console and file log
    # logger.info('#### End testing: '+ message_test)
    # logger.info(f"==>> All  psnr_score_ctr: {psnr_score_ctr:.2f} , ssim_score_ctr: {ssim_score_ctr:.4f} , psnr_score_ave: {psnr_score_ave:.2f} , ssim_score_ave: {ssim_score_ave:.4f} ")


    # if rank <= 0:
    #     logger.info('Saving the final model.')
    #     model.save('latest')
    #     logger.info('End of training.')


if __name__ == '__main__':
    main()
