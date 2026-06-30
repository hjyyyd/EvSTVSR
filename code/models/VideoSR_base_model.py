import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss
from pdb import set_trace as bp
# from ..utils.util import calculate_psnr, calculate_ssim

from utils.ssim import ssim

import models.modules.Sakuya_arch as Sakuya_arch
####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LIIF':
        #! FORWARD
        FORWARD=None
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'])   
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


logger = logging.getLogger('inrlog')

def g(*args, **kwargs):  # debug no-op (dmfq removed for release)
    pass

class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        #! core

        opt_net = opt['network_G']
        # which_model = opt_net['which_model_G']

        #! FORWARD
        FORWARD=None
        self.netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                back_RBs=opt_net['back_RBs']) 
        

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.net_opt = opt['network_G']
        self.net_base = self.net_opt['which_model_G']
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            # self.l_pix_w = train_opt['pixel_weight']
            # self.l_flow_w = train_opt['flow_weight']

            self.cbloss = CharbonnierLoss().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        '''
        data['LQs']
               ([bs, 2, 3, h, w])
        data['GT']:List len=9, 
            [0]([bs, 1, 3, hh, ww]) 
        '''
        self.var_L = data['LQs'].to(self.device)
        data['spatial_upscale'] = int(data['spatial_upscale'][0])
        # print(f"==>> type(data['spatial_upscale']): {type(data['spatial_upscale'])}")  # type: int
        self.spatial_upscale = data['spatial_upscale']
        # print(f"==>> data['spatial_upscale']: {data['spatial_upscale']}")
        # print(f"==>> In func feed_data, spatial_upscale: {self.spatial_upscale}")
        logger.debug(f"==>> self.var_L.shape: {self.var_L.shape}")
        
        self.sce_name_l = data['sce_name_l']       # List[List:len=bs]:len=9     
        self.gt_png_name_l = data['gt_png_name_l'] # List[List:len=bs]:len=9
        self.lq_png_name_l = data['lq_png_name_l']

        self.bs, _,_,  self.h,  self.w  = data['LQs'].shape   # [bs, 2, 3, h,  w]
        self.bs, _,_, self.hh, self.ww  = data['GT'][0].shape # [bs, 1, 3, hh, ww]


        self.var_flow_l = [item.to(self.device) for item in data['Flow']] 
        # self.var_evt_LQs = data['evt_LQs'].to(self.device)

        self.var_evt_LQs_0t_l = [item.to(self.device) for item in data['ev_0t_voxel_LQ_l']]
        self.var_evt_LQs_t1_l = [item.to(self.device) for item in data['ev_t1_voxel_LQ_l']] 
        self.var_evt_LQs_01_l = [item.to(self.device) for item in data['ev_01_voxel_LQ_l']] 

        if ('time' in data.keys()) and self.net_base == 'LIIF':
            self.times_l = [t_.to(self.device) for t_ in data['time']]
        else:
            self.times_l = None
        if 'scale' in data.keys():
            self.scale = data['scale']
        else:
            self.scale = None
        
        logger.debug(f"==>> self.scale: {self.scale}")


        # if 'test' in data.keys():
        #     self.testmode = data['test']
        # else:
        #     self.testmode = False
        if need_GT:
            self.real_H_l = [item.to(self.device) for item in data['GT']]    # List:len=9 ([1, 1, 3, 352, 640])             data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0
    

    def optimize_parameters(self, current_epoch,current_step):
        self.optimizer_G.zero_grad()
        
        if current_epoch<=50:
            FORWARD=None
            self.fake_H, self.preds_flow_t0_l,self.preds_flow_t1_l,self.imgt_0bwarped_l,self.imgt_1bwarped_l, self.loss_flow_smooth = \
                self.netG(self.var_L,self.real_H_l,self.spatial_upscale,self.var_evt_LQs_0t_l,self.var_evt_LQs_t1_l, self.var_evt_LQs_01_l, self.var_flow_l, self.times_l, self.scale,mode='flow')
        else:
            self.fake_H, self.preds_flow_t0_l,self.preds_flow_t1_l,self.imgt_0bwarped_l,self.imgt_1bwarped_l, self.loss_flow_smooth = \
                self.netG(self.var_L,self.real_H_l,self.spatial_upscale,self.var_evt_LQs_0t_l,self.var_evt_LQs_t1_l, self.var_evt_LQs_01_l, self.var_flow_l, self.times_l, self.scale,mode='joint')



        # self.var_flow:  ([1, 2, 2, 256, 256])
        l1loss = nn.L1Loss(reduction='sum').to(self.device)
        # flow_gt = torch.concat([self.var_flow_l[0][:,0,:,:,:], self.var_flow_l[0][:,1,:,:,:]],axis=1) #  ([1, 4, 256, 256])?
        # print(f"==>> flow_gt.shape: {flow_gt.shape}") #([1, 4, 256, 256])
        
        #* weight
        self.l_pix_w =1.0
        self.l_flow_w = 0.1
        
        self.l_photo_w = 1
        self.l_edge_w = 10
        if self.times_l is None:
            loss_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H_l[0])
        else:
            loss_pix = 0
            loss_photometric = 0
            loss_edge_aware = 0
            # loss_flow=0
            for idx in range(len(self.times_l)):
                # logger.info(f"==>> in optimize_parameters,len(self.times_l) : {len(self.times_l)}")
                # logger.info(f"==>> in optimize_parameters,range(len(self.times_l) ,  idx: {idx}")
                loss_pix += self.cri_pix(self.fake_H[idx], self.real_H_l[0][:, idx])   # self.real_H_l[0][:, idx]: 2,3,128,128 0.0~1.0
                # print(f"==>> self.flow_pred_l[idx].shape: {self.flow_pred_l[idx].shape}") # ([1, 4, 256, 256])
                # loss_flow +=  l1loss(self.flow_pred_l[idx], flow_gt)
            # self.flow_pred_l len:1  shape b2hw
            # self.var_flow len:

                # add ssim loss
                loss_ssim = 1.0 - ssim(self.fake_H[idx], self.real_H_l[0][:, idx])
                # print(f"==>> self.real_H_l: ")
                # g(self.real_H_l)  # len 1 ([2, 1, 3, 256, 256])
                # print(f"==>> self.imgt_0bwarped_l: ") 
                # g(self.imgt_0bwarped_l)   # len: 1 ([2, 3, 256, 256])
                loss_photometric += self.cbloss(self.real_H_l[0][:, idx], self.imgt_0bwarped_l[0][:,:,:,:]) + \
                    self.cbloss(self.real_H_l[0][:, idx], self.imgt_1bwarped_l[0][:,:,:,:])
                
                #todo
                loss_edge_aware  += self.loss_flow_smooth

        loss_pix  = self.l_pix_w  *  loss_pix/(self.bs*self.hh*self.ww)

        loss_photometric = self.l_photo_w  *  loss_photometric/(self.bs*self.hh*self.ww)
        loss_edge_aware = self.l_edge_w * loss_edge_aware.mean()  # [bs,] -> float

        # loss_flow = self.l_flow_w * loss_flow/(flow_gt.shape[-2]*flow_gt.shape[-1])
        
        # print(f"==>> loss_pix: {loss_pix}")
        # g(loss_pix)
        # loss_photometric = 1
        #! LOSS
        LOSS=None
        if current_epoch<=50:
            loss_all =  loss_photometric +  loss_edge_aware
            # print(f"==>> loss_all: {loss_all}")
            # g(loss_all)
            # print(f"==>> loss_photometric: {loss_photometric}")
            # g(loss_photometric)
            # print(f"==>> loss_edge_aware: {loss_edge_aware}")
            # g(loss_edge_aware)
            # set log
            self.log_dict['l_photometric'] = loss_photometric.item()
            self.log_dict['l_edge_aware'] = loss_edge_aware.item()
            self.log_dict['loss_all'] = loss_all.item()
        else:
            #! frozen flownet
            for param in self.netG.model_cbmnet.net.flownet.parameters():
                param.requires_grad = False

            loss_all =  loss_pix + loss_ssim# + loss_flow

            self.log_dict['l_pix'] = loss_pix.item()
            self.log_dict['l_ssim'] = loss_ssim.item()
            self.log_dict['loss_all'] = loss_all.item()

        # l_pix.backward()
        loss_all.backward()
        # print("Training Loss: ", l_pix.item())
        self.optimizer_G.step()


        
        # self.log_dict['loss_flow'] = loss_flow.item()


    def test(self, output=False):
        self.netG.eval()
        with torch.no_grad():
            if self.times_l is None:
                self.fake_H, self.preds_flow_t0_l,self.preds_flow_t1_l,self.imgt_0bwarped_l,self.imgt_1bwarped_l, self.loss_flow_smooth = \
                    self.netG(self.var_L,self.real_H_l,self.spatial_upscale,self.var_evt_LQs_0t_l,self.var_evt_LQs_t1_l, self.var_evt_LQs_01_l, self.var_flow_l)
            elif self.net_base == 'LIIF': #! entrance
                self.fake_H, self.preds_flow_t0_l,self.preds_flow_t1_l,self.imgt_0bwarped_l,self.imgt_1bwarped_l, self.loss_flow_smooth = \
                    self.netG(self.var_L,self.real_H_l,self.spatial_upscale,self.var_evt_LQs_0t_l,self.var_evt_LQs_t1_l, self.var_evt_LQs_01_l, self.var_flow_l, self.times_l, self.scale, mode='joint', is_test=True)
        self.netG.train()
        if output == True:
            return self.fake_H

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L         # self.var_L.detach()[0].float().cpu()
        out_dict['LQs_ev_0t_l'] = self.var_evt_LQs_0t_l         # self.var_L.detach()[0].float().cpu()
        out_dict['LQs_ev_t1_l'] = self.var_evt_LQs_t1_l         # self.var_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H_l      # self.real_H_l.detach()[0].float().cpu()
        out_dict['Fake'] = self.fake_H      # self.fake_H.detach()[0].float().cpu()
        out_dict['sce_name_l'] = self.sce_name_l
        out_dict['gt_png_name_l'] = self.gt_png_name_l
        out_dict['lq_png_name_l'] = self.lq_png_name_l
        out_dict['preds_flow_t0_l'] = self.preds_flow_t0_l
        out_dict['preds_flow_t1_l'] = self.preds_flow_t1_l


        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        print(f"==>> load_path_G: {load_path_G}")
        if load_path_G is not None:
            if self.rank <= 0:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
