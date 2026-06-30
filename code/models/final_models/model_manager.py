import torch.nn as nn
import torch
import os
from models.final_models.submodules import *
from math import ceil
import importlib
import models.final_models.ours as mod
def g(*args, **kwargs):  # debug no-op (dmfq removed for release)
    pass

class OurModel(nn.Module): # object  #! y 要把父类改成 nn.Module 才能有可学习参数
    def __init__(self, voxel_num_bins, flow_debug):
        super(OurModel, self).__init__()
        # define network
        self.voxel_num_bins = voxel_num_bins

        # batch
        self.batch = {}
        # flow debug -> default is fals
        self.flow_debug = bool(flow_debug)

        self.net = mod.EventInterpNet(self.voxel_num_bins, self.flow_debug)
    
    # def initialze(self, model_folder, model_name):
    #     mod = importlib.import_module('models.' + model_folder + '.' + model_name)
    #     self.net = mod.EventInterpNet(self.voxel_num_bins, self.flow_debug)
    
    def warp(self, x, flo):
        '''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask, mask
    
    # def cuda(self):
    #     self.net.cuda()
    
    # def train(self):
    #     self.net.train()
    
    # def eval(self):
    #     self.net.eval()
    
    # def use_multi_gpu(self): # data parallel
    #     self.net = nn.DataParallel(self.net)

    def set_input(self, sample):
        _, _,  height, width = sample['clean_image_first'].shape
        # image input
        self.batch['image_input0'] = sample['clean_image_first'].float()
        self.batch['image_input1'] = sample['clean_image_last'].float()
        self.batch['imaget_input'] = sample['clean_middle'].float()
        # event voxel grid
        self.batch['event_input_t0'] = sample['voxel_grid_t0'].float()
        self.batch['event_input_0t'] = sample['voxel_grid_0t'].float()
        self.batch['event_input_t1'] = sample['voxel_grid_t1'].float()
    
    def set_test_input_old(self, sample):
        # shape configuration
        B, _, H, W = sample['clean_image_first'].shape
        H_ = ceil(H/64)*64
        W_ = ceil(W/64)*64
        C1 = torch.zeros((B, 3, H_, W_)).cuda()
        C2 = torch.zeros((B, 3, H_, W_)).cuda()
        Cgt = torch.zeros((B, 3, H_, W_)).cuda()
        self.batch['image_input0_org'] = sample['clean_image_first']
        self.batch['image_input1_org'] = sample['clean_image_last']
        C1[:, :, 0:H, 0:W] = sample['clean_image_first']
        C2[:, :, 0:H, 0:W] = sample['clean_image_last']
        # image input
        self.batch['image_input0'] = C1
        self.batch['image_input1'] = C2
        self.batch['imaget_input'] = Cgt
        # event input
        Vt0 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        Vt1 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        V0t = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        # print(f"==>> Vt0.shape: {Vt0.shape}")
        # print(f"==>> sample['voxel_grid_t0'].shape: {sample['voxel_grid_t0'].shape}")
        Vt0[:, :, 0:H, 0:W] = sample['voxel_grid_t0']
        Vt1[:, :, 0:H, 0:W] = sample['voxel_grid_t1']
        V0t[:, :, 0:H, 0:W] = sample['voxel_grid_0t']
        # event voxel grid
        self.batch['event_input_t0'] = Vt0
        self.batch['event_input_0t'] = V0t
        self.batch['event_input_t1'] = Vt1
        # parameter configuations
        self.H_org = H
        self.W_org = W

    def set_test_input(self, sample):
        # shape configuration
        B, _, H, W = sample['clean_image_first'].shape
        H_ = ceil(H/64)*64
        W_ = ceil(W/64)*64
        C1 = torch.zeros((B, 3, H_, W_)).cuda()
        C2 = torch.zeros((B, 3, H_, W_)).cuda()
        Cgt = torch.zeros((B, 3, H_, W_)).cuda()
        self.batch['image_input0_org'] = sample['clean_image_first']
        self.batch['image_input1_org'] = sample['clean_image_last']
        C1[:, :, 0:H, 0:W] = sample['clean_image_first']
        C2[:, :, 0:H, 0:W] = sample['clean_image_last']
        # image input
        self.batch['image_input0'] = C1 #[:, :, 0:192, 0:320]
        self.batch['image_input1'] = C2 #[:, :, 0:192, 0:320]
        self.batch['imaget_input'] = Cgt# [:, :, 0:192, 0:320]
        # event input
        Vt0 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        Vt1 = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        V0t = torch.zeros((B, self.voxel_num_bins, H_, W_)).cuda()
        # print(f"==>> Vt0.shape: {Vt0.shape}")
        # print(f"==>> sample['voxel_grid_t0'].shape: {sample['voxel_grid_t0'].shape}")
        Vt0[:, :, 0:H, 0:W] = sample['voxel_grid_t0']
        Vt1[:, :, 0:H, 0:W] = sample['voxel_grid_t1']
        V0t[:, :, 0:H, 0:W] = sample['voxel_grid_0t']
        # event voxel grid
        self.batch['event_input_t0'] = Vt0 #[:, :, 0:192, 0:320]  # 1_4 :  240, 360 - > 192(64*3) 256(64*4)
        self.batch['event_input_0t'] = V0t #[:, :, 0:192, 0:320]
        self.batch['event_input_t1'] = Vt1 #[:, :, 0:192, 0:320]
        # parameter configuations
        self.H_org = H
        self.W_org = W

    def forward_joint_test(self,spatial_upscale,mode='joint',img_gt_b3hw=None):

        OUT1=None
        #! net
        FORWARD=None

        if mode == 'flow':
            self.batch['clean_middle_est'], \
            self.batch['OF_est_t0'], \
            self.batch['OF_est_t1'],\
            self.batch['imgt_0bwarped'], \
            self.batch['imgt_1bwarped'], \
            self.batch['loss_flow_smooth'] = self.net(self.batch, spatial_upscale, mode='flow')
        
            # self.batch['imgt_0bwarped'] = self.batch['imgt_0bwarped'][..., 0:self.H_org,0:self.W_org]
            # self.batch['imgt_1bwarped'] = self.batch['imgt_1bwarped'][..., 0:self.H_org,0:self.W_org]
            # self.batch['loss_flow_smooth'] = self.batch['loss_flow_smooth']


        elif mode == 'joint':
            self.batch['clean_middle_est'], \
            self.batch['OF_est_t0'], \
            self.batch['OF_est_t1'],\
            self.batch['imgt_0bwarped'], \
            self.batch['imgt_1bwarped'], \
            self.batch['loss_flow_smooth'] = self.net(self.batch, spatial_upscale, mode='joint')


            OUT2=None
            # spatial_upscale =  4 # 2 #!
            #* 这里的输出是  self.batch['clean_middle_est'][0] : ([1, 3,  768, 1280])    ([1, 3, 384, 640])   ([1, 3, 192, 320])       192/32=6,  704/4=176 176/32=5
            # self.batch['clean_middle_est'] = self.batch['clean_middle_est'][0][..., 0:self.H_org*spatial_upscale,0:self.W_org*spatial_upscale]
            # self.batch['OF_est_t0'] = self.batch['OF_est_t0'][0][..., 0:self.H_org,0:self.W_org]
            # self.batch['OF_est_t1'] = self.batch['OF_est_t1'][0][..., 0:self.H_org,0:self.W_org]


            # print('9'*10)
            # g(self.batch['clean_middle_est'])

        
    def load_model(self, state_dict):
        self.net.load_state_dict(state_dict)
        print('load model')