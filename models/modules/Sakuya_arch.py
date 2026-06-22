'''
The code is modified from the implementation of Zooming Slow-Mo:
https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/models/modules/Sakuya_arch.py
'''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell
# try:
#     from models.modules.DCNv2_latest.dcn_v2 import DCN_sep
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')

import os


from models.final_models.model_manager import OurModel
def g(*args, **kwargs):  # debug no-op (dmfq removed for release)
    pass



from pdb import set_trace as bp
from models.modules.SIREN import Siren, warp_Siren
from models.modules.warplayer import warpgrid
import flow_vis


import logging
logger = logging.getLogger('inrlog')


# import torchshow as ts
from skimage.metrics import peak_signal_noise_ratio as ski_calc_psnr
import numpy as np

# from global_config import input_vis,flow_vis_bool
#tag abc
# def cal_err_tc_2hw3_to_1hw(tensor1,tensor2):
#     diff = torch.sub(tensor1, tensor2)
#     # 在 dim=2 上求平方和
#     squared_sum = torch.sum(diff**2, dim=2)
#     return squared_sum






from models.modules.submodules_y import event_encoder


#tag model
class LunaTokis(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):    # test (64, 6, 8, 5, 40)
        super(LunaTokis, self).__init__()

        self.model_cbmnet = OurModel(voxel_num_bins=16,  flow_debug=False)
        # self.model_cbmnet.initialze(model_folder='final_models', model_name='ours')

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        
    #todo x_ev_01_l
    def forward(self, imgs_in, img_gt_l, spatial_upscale, x_ev_0t_l, x_ev_t1_l, x_ev_01_l, imgs_flow_l, times_l=None, scale=None, is_test=False,mode='joint'):
        '''
        param:
            imgs_in:        bs,2,3,h,w
                ([4, 2, 3, 120, 120])     0.0~1.0
            img_gt_l:       bs,1,3,hh,ww
                List: len same to times: 1
                ([4, 1, 3, 480, 480])
            x_ev_0t_l:      bs,1,16,h,w
                List, len same to times: 1
                ([4, 1, 16, 120, 120])
            x_ev_t1_l:      bs,1,16,h,w   
                List, len same to times: 1
                ([4, 1, 16, 120, 120])
            x_ev_01_l:      bs,16,1,h,w
                List, len same to times: 1
                ([4, 1, 16, 120, 120])

            times_l:
                for train:  list, len=1
                for test:   list, len=8：  0, 0.125,..., 0.875
                    1
        out:
            preds_l: len same to times: 1
                        ([4, 3, 120, 120])
        '''

        is_train = not is_test
        logger.debug(f"==>> imgs_in.shape: {imgs_in.shape}")  # bs, 2, 3,128,128
        logger.debug(f"==>> img_gt_l[0].shape: {img_gt_l[0].shape}") #[c] 1,3,128,128
        logger.debug(f"==>> x_ev_t1_l[0].shape: {x_ev_t1_l[0].shape}") # [c] 16,1,128,128
        # if is_train: #
        FORWARD=None

        # preds_l,preds_flow_t0_l,preds_flow_t1_l, imgt_0bwarped_l, imgt_1bwarped_l, loss_flow_smooth = self.cbmnet_forward(imgs_in, img_gt_l,spatial_upscale, x_ev_0t_l, x_ev_t1_l,  imgs_flow_l, times_l, scale=None, is_test=False,mode=mode)
        # logger.debug(f"==>> times_l: {times_l}")
        '''
        imgs_in:          ([1, 2, 3, 64, 64])
        img_gt_l:   
            list: len same to times
                    ([1, 1, 3, 256, 256])
        x_ev_0t:    ([1, 16, 1, 64, 64])
        x_ev_t1:    ([1, 16, 1, 64, 64])

        times:
            for train:  list, len=1
            for test:   list, len=8：  0, 0.125,..., 0.875
                1
        '''
        # patch-wise evaluation
        # iter_idx = 0
        # h_size_patch_testing = 640   # 640*2-305=975
        # h_overlap_size = 305
        # w_size_patch_testing = 896   # 896*2-352=1440
        # w_overlap_size = 352

        height = imgs_in.shape[-2] #128  #  # 640*2-305=975 #!
        h0 = 0
        width = imgs_in.shape[-1] # 128   # 896*2-352=1440
        w0 = 0

        sample = {}

        frame1 = imgs_in[:,0,:,:,:] #[:, :, 0:192, 0:320] # bchw 1 3 240 360
        frame3 = imgs_in[:,1,:,:,:] #[:, :, 0:192, 0:320] # bchw 1 3 240 360
        sample['clean_image_first'] = frame1    # bs,3,h,w
        sample['clean_image_last'] = frame3     # bs,3,h,w

        preds_l=[]
        preds_flow_t0_l = []
        preds_flow_t1_l = []
        imgt_0bwarped_l= []
        imgt_1bwarped_l = []
        for c in range(len(times_l)):  # 1?)
            
            img_gt_b3hw = img_gt_l[c][:,0,:,:,:]  # ([bs, 3, 256, 256])?

            voxel_0t = x_ev_0t_l[c][:,0,:,:,:]  # bs,16,h,w     # 1 16 240 360
            # print(f"==>> voxel_0t.shape: {voxel_0t.shape}")
            voxel_t1 = x_ev_t1_l[c][:,0,:,:,:]  # bs,16,h,w     # 1 16 240 360
            # print(f"==>> voxel_t1.shape: {voxel_t1.shape}")
            voxel_t0 = - voxel_0t.flip(dims=(1,))               # 1 16 240 360
            # print(f"==>> voxel_t0.shape: {voxel_t0.shape}")
            sample['voxel_grid_0t'] = voxel_0t #[:, :, 0:192, 0:320]
            sample['voxel_grid_t1'] = voxel_t1 #[:, :, 0:192, 0:320]
            sample['voxel_grid_t0'] = voxel_t0 #[:, :, 0:192, 0:320]

            bs, cs, height, width  = frame1.shape

            h_stride = height - h0 
            w_stride = width - w0   
            
            # h_idx_list = list(range(0, height-height, h_stride)) + [max(0, height-height)] 
            # w_idx_list = list(range(0, width-width, w_stride)) + [max(0, width-width)]
            # output
            # print(f"==>> spatial_upscale: {spatial_upscale}")
            # spatial_upscale =  spatial_upscale[0] # 4 #todo
            
            hh,ww = int(height*spatial_upscale), int(width*spatial_upscale)
            E = torch.zeros(bs, cs, hh, ww).cuda()
            # print(f"==>> E.shape: {E.shape}")
            W_ = torch.zeros_like(E).cuda()
            input_keys = ['clean_image_first', 'clean_image_last', 'voxel_grid_0t', 'voxel_grid_t1', 'voxel_grid_t0']
            not_overlap_border = True

            # for 0 in h_idx_list:  # 其实 是0
            #     for 0 in w_idx_list: # 其实是 0
            _sample = {}
            for input_key in input_keys:
                _sample[input_key] = sample[input_key][..., 0:0+height, 0:0+width]
            self.model_cbmnet.set_test_input(_sample)
            #!
            FORWARD=None
            self.model_cbmnet.forward_joint_test(spatial_upscale,mode=mode,img_gt_b3hw=img_gt_b3hw)
            OUT=None
            out_patch = self.model_cbmnet.batch['clean_middle_est']
            out_flow_t0_patch = self.model_cbmnet.batch['OF_est_t0']    #*
            out_flow_t1_patch = self.model_cbmnet.batch['OF_est_t1']   # h,w,
            
            imgt_0bwarped = self.model_cbmnet.batch['imgt_0bwarped']
            imgt_1bwarped = self.model_cbmnet.batch['imgt_1bwarped']
            loss_flow_smooth = self.model_cbmnet.batch['loss_flow_smooth']
            # print(f"==>> out_patch.shape: {out_patch.shape}")
            # out_patch_mask = torch.ones_like(out_patch)
            if not_overlap_border:
                pass
            E.add_(out_patch)
            # W_.add_(out_patch_mask)
            
            # clean_middle_tc_output = E.div_(W_) # b3hw
            clean_middle_tc_output = E
            # print('a'*10)
            # get_info(clean_middle_tc_output)
            # clean_middle_tc_output /= 255.  #!
            
            preds_l.append(clean_middle_tc_output)
            preds_flow_t0_l.append(out_flow_t0_patch)
            preds_flow_t1_l.append(out_flow_t1_patch)

            imgt_0bwarped_l.append(imgt_0bwarped)
            imgt_1bwarped_l.append(imgt_1bwarped)
        return preds_l, preds_flow_t0_l,preds_flow_t1_l, imgt_0bwarped_l, imgt_1bwarped_l, loss_flow_smooth

