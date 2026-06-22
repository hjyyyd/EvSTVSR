from models.final_models.submodules import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import conv
from correlation_package.correlation import Correlation 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import reduce, lru_cache
import torch.nn.functional as tf
from torch.autograd import Variable
from einops import rearrange
import math
import numbers
import collections

import torch.nn.functional as F

def g(*args, **kwargs):  # debug no-op (dmfq removed for release)
    pass


from models.modules.SIREN import Siren


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        # print(f"==>> v0: {v0}")   # int
        if isinstance(r, float):
            # print(f"==>> r: {r}")
            seq = v0 + r + (2 * r) * torch.arange(n).float()
        elif isinstance(r, torch.Tensor):
            # print(f"==>> r.device: {r.device}")
            # print(f"==>> r: {r}")
            device = r.device
            seq = v0 + r + (2 * r) * torch.arange(n,device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class encoder_event_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=1)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13


class encoder_event_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13


class encoder_image_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_image_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one(num_chs[3], num_chs[4], stride=2)
    
    def forward(self, image):
        x = self.conv1(image)
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)
        return f1, f2, f3

def upsample2d(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def upsample2d_hw(inputs, h, w, mode="bilinear"):
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class DenseBlock(nn.Module):
    def __init__(self, ch_in):
        super(DenseBlock, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out



class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in=64, f_channels=(128, 128, 96, 64, 32, 32), ch_out=2):
        super(FlowEstimatorDense, self).__init__()
        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1
        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], axis=1)
        x2 = torch.cat([self.conv2(x1), x1], axis=1)
        x3 = torch.cat([self.conv3(x2), x2], axis=1)
        x4 = torch.cat([self.conv4(x3), x3], axis=1)
        x5 = torch.cat([self.conv5(x4), x4], axis=1)
        x_out = self.conv_last(x5)
        return x5, x_out

class Tfeat_RefineBlock(nn.Module):
    def __init__(self, ch_in_frame, ch_in_event, ch_in_frame_prev, prev_scale=False):
        super(Tfeat_RefineBlock, self).__init__()
        if prev_scale:
            nf = int((ch_in_frame*2+ch_in_event+ch_in_frame_prev)/4)
        else:
            nf = int((ch_in_frame*2+ch_in_event)/4)
        self.conv_refine = nn.Sequential(conv1x1(4*nf, nf), nn.ReLU(), conv3x3(nf, 2*nf), nn.ReLU(), conv_resblock_one(2*nf, ch_in_frame))
    
    def forward(self, x):
        x1 = self.conv_refine(x)
        return x1

def rescale_flow(flow, width_im, height_im):
    u_scale = float(width_im / flow.size(3))
    v_scale = float(height_im / flow.size(2))
    u, v = flow.chunk(2, dim=1)
    u = u_scale*u
    v = v_scale*v
    return torch.cat([u, v], dim=1)


class FlowNet(nn.Module):
    def __init__(self, md=4, tb_debug=False):
        super(FlowNet, self).__init__()
        ## argument
        self.tb_debug = tb_debug
        # flow scale
        self.flow_scale = 20
        num_chs_frame = [3, 16, 32, 64, 96]
        num_chs_event = [16, 16, 32, 64, 128]
        num_chs_event_image = [16, 16, 16, 32, 64]
        ## for event-level flow
        self.encoder_event = encoder_event_flow(num_chs_event)
        ## for image-level flow
        self.encoder_image_flow = encoder_image_for_image_flow(num_chs_frame)
        self.encoder_image_flow_event = encoder_event_for_image_flow(num_chs_event_image)
        ## leaky relu
        self.leakyRELU = nn.LeakyReLU(0.1)
        ## correlation channel value
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        ## correlation channel value
        nd = (2*md+1)**2
        self.corr_refinement = nn.ModuleList([DenseBlock(nd+num_chs_frame[-1]+2), 
        DenseBlock(nd+num_chs_frame[-2]+2), 
        DenseBlock(nd+num_chs_frame[-3]+2),
        DenseBlock(nd+num_chs_frame[-4]+2)
        ])
        self.decoder_event = nn.ModuleList([conv_resblock_one(num_chs_event[-1], num_chs_event[-1]),
                                            conv_resblock_one(num_chs_event[-2]+ num_chs_event[-1]+2, num_chs_event[-2]),
                                            conv_resblock_one(num_chs_event[-3]+ num_chs_event[-2]+2, num_chs_event[-3])])
        self.predict_flow = nn.ModuleList([conv3x3_leaky_relu(num_chs_event[-1], 2),
                                           conv3x3_leaky_relu(num_chs_event[-2], 2),
                                           conv3x3_leaky_relu(num_chs_event[-3], 2)])
        self.conv_frame = nn.ModuleList([conv3x3_leaky_relu(num_chs_frame[-2], 32),
                                         conv3x3_leaky_relu(num_chs_frame[-3], 32)])
        self.conv_frame_t = nn.ModuleList([conv3x3_leaky_relu(num_chs_frame[-2], 32),
                                           conv3x3_leaky_relu(num_chs_frame[-3], 32)])
        self.flow_fusion_block = FlowEstimatorDense(32*3+4, (32, 32, 32, 16, 8), 1) 
        self.feat_t_refinement = nn.ModuleList([Tfeat_RefineBlock(num_chs_frame[-1], num_chs_event_image[-1]*2, None, prev_scale=False),
                                                Tfeat_RefineBlock(num_chs_frame[-2], num_chs_event_image[-2]*2, num_chs_frame[-1], prev_scale=True),
                                                Tfeat_RefineBlock(num_chs_frame[-3], num_chs_event_image[-3]*2, num_chs_frame[-2], prev_scale=True),
                                                ])


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

    def normalize_features(self, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        # Compute feature statistics.
        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, axis=axes, keepdims=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            statistics['mean'] = ([torch.mean(F.stack(statistics['mean'], axis=0), axis=(0, ))] * len(feature_list))
            statistics['var'] = ([torch.var(F.stack(statistics['var'], axis=0), axis=(0, ))] * len(feature_list))

        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]
        # Center and normalize features.
        if center:
            feature_list = [f - mean for f, mean in zip(feature_list, statistics['mean'])]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]
        return feature_list

    def forward(self, batch):
        # F for frame feature
        # E for event feature
        ### encoding
        ## image feature
        # feature pyramid
        F0_pyramid = self.encoder_image_flow(batch['image_input0'])[::-1] # ([bs, 3, 64, 64])
        F1_pyramid = self.encoder_image_flow(batch['image_input1'])[::-1] # ([bs, 3, 64, 64])
        E_0t_pyramid = self.encoder_image_flow_event(batch['event_input_0t'])[::-1]
        E_t1_pyramid = self.encoder_image_flow_event(batch['event_input_t1'])[::-1]
        # encoder event
        E_t0_pyramid_flow = self.encoder_event(batch['event_input_t0'])[::-1]
        E_t1_pyramid_flow = self.encoder_event(batch['event_input_t1'])[::-1]
        ### decoding optical flow
        ## level 0
        flow_t0_out_dict, flow_t1_out_dict, flow_t0_dict, flow_t1_dict = [], [], [], []
        ## event flow and image flow
        if self.tb_debug:
            event_flow_dict, image_flow_dict, fusion_flow_dict, mask_dict = [], [], [], []
        for level, (E_t0_flow, E_t1_flow, E_0t, E_t1, F0, F1) in enumerate(zip(E_t0_pyramid_flow, E_t1_pyramid_flow, E_0t_pyramid, E_t1_pyramid, F0_pyramid, F1_pyramid)):
            if level==0:
                ## event flow generation
                feat_t0_ev = self.decoder_event[level](E_t0_flow)
                feat_t1_ev = self.decoder_event[level](E_t1_flow)
                flow_event_t0 = self.predict_flow[level](feat_t0_ev)
                flow_event_t1 = self.predict_flow[level](feat_t1_ev)
                ## fusion flow(scale == 0)
                flow_fusion_t0 = flow_event_t0
                flow_fusion_t1 = flow_event_t1
                ## t feature
                feat_t_in = torch.cat((F0, F1, E_0t, E_t1), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)
            else:
                ## feat t
                upfeat0_t = upsample2d(feat_t, F0)
                feat_t_in = torch.cat((upfeat0_t, F0, F1, E_0t, E_t1), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)
                #### event-based optical flow
                ## event flow generation
                upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level-1], E_t0_flow), E_t0_flow.size(3), E_t0_flow.size(2))
                upflow_t1 = rescale_flow(upsample2d(flow_t1_out_dict[level-1], E_t1_flow), E_t1_flow.size(3), E_t1_flow.size(2))
                # upsample feat_t0
                feat_t0_ev_up = upsample2d(feat_t0_ev, E_t0_flow)
                feat_t1_ev_up = upsample2d(feat_t1_ev, E_t1_flow)
                # decoder event
                flow_t0_ev_up = rescale_flow(upsample2d(flow_event_t0, E_t0_flow), E_t0_flow.size(3), E_t0_flow.size(2))
                flow_t1_ev_up = rescale_flow(upsample2d(flow_event_t1, E_t1_flow), E_t1_flow.size(3), E_t1_flow.size(2))
                feat_t0_ev = self.decoder_event[level](torch.cat((E_t0_flow, feat_t0_ev_up, flow_t0_ev_up ), dim=1))
                feat_t1_ev = self.decoder_event[level](torch.cat((E_t1_flow, feat_t1_ev_up, flow_t1_ev_up), dim=1))
                ## project flow
                flow_event_t0_ = self.predict_flow[level](feat_t0_ev)
                flow_event_t1_ = self.predict_flow[level](feat_t1_ev)
                ## fusion flow
                flow_event_t0 = flow_t0_ev_up + flow_event_t0_
                flow_event_t1 = flow_t1_ev_up + flow_event_t1_
                # flow rescale
                down_evflow_t0 = rescale_flow(upsample2d(flow_event_t0, F0), F0.size(3), F0.size(2))
                down_evflow_t1 = rescale_flow(upsample2d(flow_event_t1, F1), F1.size(3), F1.size(2))
                down_upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level-1], F0), F0.size(3), F0.size(2))
                down_upflow_t1 = rescale_flow(upsample2d(flow_t1_out_dict[level-1], F1), F1.size(3), F1.size(2))
                ## warping with event flow and fusion flow
                F0_re = self.conv_frame[level-1](F0)
                F0_up_warp_ev = self.warp(F0_re, self.flow_scale*down_evflow_t0)
                F0_up_warp_frame = self.warp(F0_re, self.flow_scale*down_upflow_t0)
                F1_re = self.conv_frame[level-1](F1)
                F1_up_warp_ev = self.warp(F1_re, self.flow_scale*down_evflow_t1)
                F1_up_warp_frame = self.warp(F1_re, self.flow_scale*down_upflow_t1)
                Ft_up = self.conv_frame_t[level-1](feat_t)
                ## flow fusion
                _, out_fusion_t0 = self.flow_fusion_block(torch.cat((F0_up_warp_ev, F0_up_warp_frame, Ft_up, down_evflow_t0, down_upflow_t0), dim=1))
                _, out_fusion_t1 = self.flow_fusion_block(torch.cat((F1_up_warp_ev, F1_up_warp_frame, Ft_up, down_evflow_t1, down_upflow_t1), dim=1))
                mask_t0 = upsample2d(torch.sigmoid(out_fusion_t0[:, -1, : ,:])[:, None, :, :], E_t0_flow)
                mask_t1 = upsample2d(torch.sigmoid(out_fusion_t1[:, -1, :, :])[:, None, :, :], E_t1_flow)
                flow_fusion_t0 = (1-mask_t0)*upflow_t0 + mask_t0*flow_event_t0
                flow_fusion_t1 = (1-mask_t1)*upflow_t1 + mask_t1*flow_event_t1
                ## intermediate output
                if self.tb_debug:
                    event_flow_dict.append(flow_event_t0)
                    fusion_flow_dict.append(flow_fusion_t0)
                    image_flow_dict.append(upflow_t0)
                    mask_dict.append(mask_t0)
            # flow rescale
            down_flow_fusion_t0 = rescale_flow(upsample2d(flow_fusion_t0, F0), F0.size(3), F0.size(2))
            down_flow_fusion_t1 = rescale_flow(upsample2d(flow_fusion_t1, F1), F1.size(3), F1.size(2))
            # warping with optical flow
            feat10 = self.warp(F0, self.flow_scale*down_flow_fusion_t0)
            feat11 = self.warp(F1, self.flow_scale*down_flow_fusion_t1)
            # feature normalization
            feat_t_norm, feat10_norm, feat11_norm = self.normalize_features([feat_t, feat10, feat11], normalize=True, center=True, moments_across_channels=False, moments_across_images=False)
            # correlation
            corr_t0 = self.leakyRELU(self.corr(feat_t_norm, feat10_norm))
            corr_t1 = self.leakyRELU(self.corr(feat_t_norm, feat11_norm))
            # correlation refienement
            _, res_flow_t0 = self.corr_refinement[level](torch.cat((corr_t0, feat_t, down_flow_fusion_t0), dim=1))
            _, res_flow_t1 = self.corr_refinement[level](torch.cat((corr_t1, feat_t, down_flow_fusion_t1), dim=1))
            # frame-based optical flow generation
            flow_t0_frame = down_flow_fusion_t0 + res_flow_t0
            flow_t1_frame = down_flow_fusion_t1 + res_flow_t1
            ## upsampling frame-based optical flow
            upflow_t0_frame = rescale_flow(upsample2d(flow_t0_frame, flow_fusion_t0), flow_fusion_t0.size(3), flow_fusion_t0.size(2))
            upflow_t1_frame = rescale_flow(upsample2d(flow_t1_frame, flow_fusion_t1), flow_fusion_t1.size(3), flow_fusion_t1.size(2))
            ### output
            flow_t0_out_dict.append(upflow_t0_frame)
            flow_t1_out_dict.append(upflow_t1_frame)
        flow_t0_dict.append(self.flow_scale*upflow_t0_frame)
        flow_t1_dict.append(self.flow_scale*upflow_t1_frame)
        flow_t0_dict = flow_t0_dict[::-1]
        flow_t1_dict = flow_t1_dict[::-1] 
        ## final output return
        if self.tb_debug:
            return flow_t0_dict, flow_t1_dict, event_flow_dict, fusion_flow_dict, image_flow_dict, mask_dict
        else:
            return flow_t0_dict, flow_t1_dict


class frame_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(frame_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_) # ([1, 8, 640, 896])    ([4, 8, 128, 128])    
        # print(f"==>> f1.shape: {f1.shape}")
        f2 = self.conv2(f1) # ([1, 16, 320, 448])    ([4, 16, 64, 64])    
        # print(f"==>> f2.shape: {f2.shape}")
        f3 = self.conv3(f2) # ([1, 32, 160, 224])   ([4, 32, 32, 32])  
        # print(f"==>> f3.shape: {f3.shape}")
        return [f1, f2, f3]

class event_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(event_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)
    
    def forward(self, x):
        x_ =  self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

##################################################
################# Restormer #####################

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


AAAAAAAAAAA_LAYER_NORM=None
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape): # normalized_shape = dim=64*4
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x): # x: ([2, 4096, 256])
        mu = x.mean(-1, keepdim=True)   # ([2, 4096, 1])
        sigma = x.var(-1, keepdim=True, unbiased=False) # ([2, 4096, 1])
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else: # LayerNorm_type='WithBias'
            self.body = WithBias_LayerNorm(dim) # dim=64*4

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

AAAAAAAAAAA_FEEDFORWARD=None
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)   来自于论文 Restormer，  ref: cvpr2022 oral, 在图像 restoration 中好用 https://zhuanlan.zhihu.com/p/523695751
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor) # dim=256, hidden_features=680
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        # y: 深度卷积(depth-wise convolutions): 用以学习局部结构信息
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):         # x: ([2, 256, 64, 64])
        x = self.project_in(x)    # x: ([2, 680*2, 64, 64])
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # x1: ([2, 680, 64, 64])    x2: ([2, 680, 64, 64])
        # y: 门控机制(gating mechanism)
        x = F.gelu(x1) * x2     # ([2, 680, 64, 64])   和所谓的门控有关
        x = self.project_out(x) # ([2, 256, 64, 64])
        return x
AAAAAAAAAAA_CROSS_TRANSFORMERLAYER=None
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        '''
        dim=64*4, num_heads=4, bias=False
        '''
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q   = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv2 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv    = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv1_dwconv  = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.kv2_dwconv  = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim,   kernel_size=1, bias=bias)
        
    def forward(self, x, attn_kv1, attn_kv2):
        ''' # todo
        x: ([2, 256, 64, 64])   
        attn_kv1: ([2, 256, 64, 64])     
        attn_kv2: ([2, 256, 64, 64])
        '''
        b,c,h,w = x.shape

        q_ = self.q_dwconv(self.q(x))             # q_: ([2, 512, 64, 64])           self.q(x): ([2, 512, 64, 64])  
        kv1 = self.kv1_dwconv(self.kv1(attn_kv1)) # kv1:([2, 512, 64, 64])  self.kv1(attn_kv1): ([2, 512, 64, 64])
        kv2 = self.kv2_dwconv(self.kv2(attn_kv2)) # kv2:([2, 512, 64, 64])  self.kv2(attn_kv2): ([2, 512, 64, 64])
        q1,q2 = q_.chunk(2, dim=1)  # q1: ([2, 256, 64, 64])    q2: ([2, 256, 64, 64])
        k1,v1 = kv1.chunk(2, dim=1) # k1: ([2, 256, 64, 64])    v1: ([2, 256, 64, 64]) 
        k2,v2 = kv2.chunk(2, dim=1) # k2: ([2, 256, 64, 64])    v2: ([2, 256, 64, 64])   
        
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # ([2, 4, 64, 4096])

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn = (q1 @ k1.transpose(-2, -1)) * self.temperature1  # ([2, 4, 64, 64])
        attn = attn.softmax(dim=-1)                             # ([2, 4, 64, 64])
        out1 = (attn @ v1)                                      # ([2, 4, 64, 4096])

        attn = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out2 = (attn @ v2)                                      # ([2, 4, 64, 4096])
        
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   # ([2, 256, 64, 64])
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   # ([2, 256, 64, 64])
        out = torch.cat((out1, out2), dim=1)                                                        # ([2, 512, 64, 64])
        out = self.project_out(out)                                                                 # ([2, 256, 64, 64])
        return out


##########################################################################
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        '''
        dim=64*4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'
        '''
        super(CrossTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)# LayerNorm_type='WithBias'
        self.norm_kv1 = LayerNorm(dim, LayerNorm_type)
        self.norm_kv2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, attn_kv1, attn_kv2):
        '''
        x: ([2, 256, 64, 64])   
        attn_kv1: ([2, 256, 64, 64])     
        attn_kv2: ([2, 256, 64, 64])
        '''
        x = x + self.attn(self.norm1(x), self.norm_kv1(attn_kv1), self.norm_kv2(attn_kv2))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(CrossTransformerLayer, self).__init__()
        self.blocks = nn.ModuleList([CrossTransformerBlock(dim=dim, num_heads=num_heads, 
                                                          ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                                          LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
    def forward(self, x, attn_kv=None, attn_kv2=None):
        for blk in self.blocks:
            x = blk(x, attn_kv, attn_kv2)
        return x 
    
BBBBBBBBB_SELF_ATTENTIONLAYER=None
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape # ([2, 256, 64, 64])

        qkv = self.qkv_dwconv(self.qkv(x)) # ([2, 768, 64, 64])    self.qkv(x): ([2, 768, 64, 64])
        q,k,v = qkv.chunk(3, dim=1)       # q:([2, 256, 64, 64])    k:([2, 256, 64, 64])    v:([2, 256, 64, 64])
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   # q: ([2, 4, 64, 4096]) 
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   # k: ([2, 4, 64, 4096])  
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   # v: ([2, 4, 64, 4096]) 

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # ([2, 4, 64, 64])
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) # ([2, 256, 64, 64])
        return out


class Self_attention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Self_attention, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(SelfAttentionLayer, self).__init__()
        self.blocks = nn.ModuleList([Self_attention(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                                          LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x 

FFFFFFFFFFFFFFFF_UPSAMPLE=None
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.deconv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops


class Transformer(nn.Module):
    def __init__(self, unit_dim): # unit_dim=64
        super(Transformer, self).__init__()
        ## init qurey networks
        self.init_qurey_net(unit_dim)  # 
        self.init_decoder(unit_dim)
        ## last conv
        self.last_conv0 = conv3x3(unit_dim*4, 3)
        self.last_conv1 = conv3x3(unit_dim*2, 3)
        self.last_conv2 = conv3x3(unit_dim, 3)

    def init_decoder(self, unit_dim):    # unit_dim=64
        ### decoder
        ### attention k,v building (synthesis)
        self.build_kv0_syn = conv3x3_leaky_relu(unit_dim*3, unit_dim*4) # 192 -> 256
        self.build_kv1_syn = conv3x3_leaky_relu(int(unit_dim*1.5), unit_dim*2)
        self.build_kv2_syn = conv3x3_leaky_relu(int(unit_dim*0.75), unit_dim)
        ### attention k, v building (warping)
        self.build_kv0_warp = conv3x3_leaky_relu(unit_dim*3+6, unit_dim*4)
        self.build_kv1_warp = conv3x3_leaky_relu(int(unit_dim*1.5)+6, unit_dim*2)
        self.build_kv2_warp = conv3x3_leaky_relu(int(unit_dim*0.75)+6, unit_dim)
        ## level 1
        '''
        int(256*2.66)=680
        '''
        self.decoder1_1 = CrossTransformerLayer(dim=unit_dim*4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder1_2 =    SelfAttentionLayer(dim=unit_dim*4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## level 2
        self.decoder2_1 = CrossTransformerLayer(dim=unit_dim*2, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder2_2 =    SelfAttentionLayer(dim=unit_dim*2, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## level 3
        self.decoder3_1 = CrossTransformerLayer(dim=unit_dim, num_heads=4,   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        self.decoder3_2 =    SelfAttentionLayer(dim=unit_dim, num_heads=1,   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', num_blocks=2) 
        ## upsample
        self.upsample0 = Upsample(unit_dim*4, unit_dim*2)
        self.upsample1 = Upsample(unit_dim*2, unit_dim)
        ## conv after body
        self.conv_after_body0 = conv_resblock_one(4*unit_dim, 2*unit_dim)
        self.conv_after_body1 = conv_resblock_one(2*unit_dim, unit_dim)
    
    ### qurey network 
    def init_qurey_net(self, unit_dim):
        ### building query
        ## stage 1
        self.enc_conv0 = conv3x3_leaky_relu(unit_dim+6, unit_dim)  # unit_dim=64   70 -> 64
        ## stage 2
        self.enc_conv1 = conv3x3_leaky_relu(unit_dim, 2*unit_dim, stride=2)      # 64 -> 128   ,2
        ## stage 3
        self.enc_conv2 = conv3x3_leaky_relu(2*unit_dim, 4*unit_dim, stride=2)    # 128->256    ,2

    ## query buiding !! 
    def build_qurey(self, event_feature, frame_feature, warped_feature):
        '''
        event_feature   len: 3   0:([2, 32, 256, 256])   1:([2, 64, 128, 128])    2:([2, 128, 64, 64])
        frame_feature   len: 3   0:([2, 16, 256, 256])   1:([2, 32, 128, 128])    2:([2, 64, 64, 64])
        warped_feature  len: 3   0:([2, 22, 256, 256])   1:([2, 38, 128, 128])    2:([2, 70, 64, 64])
        '''
        cat_in0 = torch.cat((event_feature[0], frame_feature[0], warped_feature[0]), dim=1) # ([2, 70, 256, 256])
        Q0 = self.enc_conv0(cat_in0)    # ([2, 64, 256, 256])
        Q1 = self.enc_conv1(Q0)         # ([2, 128, 128, 128])
        Q2 = self.enc_conv2(Q1)         # ([2, 256, 64, 64])
        Q_list = [Q0, Q1, Q2]
        '''
        Q_list          len: 3   0:([2, 64, 256, 256])   1:([2, 128, 128, 128])    2:([2, 256, 64, 64])
        '''
        return Q_list # [Q0, Q1, Q2]
    
    def forward_decoder(self, Q_list, warped_feature, frame_feature, event_feature):
        '''
        Q_list:         len: 3   0:([2, 64, 256, 256])   1:([2, 128, 128, 128])    2:([2, 256, 64, 64])
        event_feature   len: 3   0:([2, 32, 256, 256])   1:([2, 64, 128, 128])    2:([2, 128, 64, 64])
        frame_feature   len: 3   0:([2, 16, 256, 256])   1:([2, 32, 128, 128])    2:([2, 64, 64, 64])
        warped_feature  len: 3   0:([2, 22, 256, 256])   1:([2, 38, 128, 128])    2:([2, 70, 64, 64])
        '''
        ## syntheis kv building
        cat_in0_syn = torch.cat((frame_feature[2], event_feature[2]), dim=1) # 64*3 # ([2, 192, 64, 64])
        attn_kv0_syn = self.build_kv0_syn(cat_in0_syn)   # 64*3        -> 64*4              # ([2, 256, 64, 64])
        cat_in1_syn = torch.cat((frame_feature[1], event_feature[1]), dim=1)
        attn_kv1_syn = self.build_kv1_syn(cat_in1_syn)   # int(64*1.5) -> 64*2              # ([2, 128, 128, 128])
        cat_in2_syn = torch.cat((frame_feature[0], event_feature[0]), dim=1)
        attn_kv2_syn = self.build_kv2_syn(cat_in2_syn)   # int(64*0.75)-> 64                # ([2, 64, 256, 256])
        ## warping kv building
        cat_in0_warp = torch.cat((warped_feature[2], event_feature[2]), dim=1)      # ([2, 198, 64, 64])
        attn_kv0_warp = self.build_kv0_warp(cat_in0_warp)   # 64*3+6   -> 64*4              # ([2, 256, 64, 64])
        cat_in1_warp = torch.cat((warped_feature[1], event_feature[1]), dim=1)
        attn_kv1_warp = self.build_kv1_warp(cat_in1_warp)   # int(64*1.5)+6  -> 64*2        # ([2, 128, 128, 128])
        cat_in2_warp = torch.cat((warped_feature[0], event_feature[0]), dim=1)
        attn_kv2_warp = self.build_kv2_warp(cat_in2_warp)   # int(64*0.75)+6 ->  64         # ([2, 64, 256, 256])
        ## out 0
        #! s: 2->0 反过来了
        _Q0 = Q_list[2]         # ([2, 256, 64, 64])
        '''
        _Q0:            ([2, 256, 64, 64])   
        attn_kv0_syn:   ([2, 256, 64, 64])     
        attn_kv0_warp:  ([2, 256, 64, 64])
        '''
        out0 = self.decoder1_1(_Q0, attn_kv0_syn, attn_kv0_warp) # ([2, 256, 64, 64])    
        out0 = self.decoder1_2(out0)   # ([2, 256, 64, 64])
        up_out0 = self.upsample0(out0) # ([2, 128, 128, 128])
        
        ## out 1
        _Q1 = Q_list[1] # ([2, 128, 128, 128])
        _Q1 = self.conv_after_body0(torch.cat((_Q1, up_out0), dim=1)) # ([2, 256, 128, 128]) -> ([2, 128, 128, 128])
        '''
        _Q1:            ([2, 128, 128, 128])  
        attn_kv1_syn:   ([2, 128, 128, 128])  
        attn_kv1_warp:  ([2, 128, 128, 128])
        '''
        out1 = self.decoder2_1(_Q1, attn_kv1_syn, attn_kv1_warp)                              # ([2, 128, 128, 128])
        out1 = self.decoder2_2(out1)        # ([2, 128, 128, 128])
        up_out1 = self.upsample1(out1)      # ([2, 64, 256, 256])
        
        ## out2
        _Q2 = Q_list[0] # ([2, 64, 256, 256])
        _Q2 = self.conv_after_body1(torch.cat((_Q2, up_out1), dim=1)) # ([2, 64, 256, 256])
        '''
        _Q2:            ([2, 64, 256, 256])
        attn_kv2_syn:   ([2, 64, 256, 256])
        attn_kv2_warp:  ([2, 64, 256, 256])
        '''
        out2 = self.decoder3_1(_Q2, attn_kv2_syn, attn_kv2_warp)      # ([2, 64, 256, 256])
        out2 = self.decoder3_2(out2)                                  # ([2, 64, 256, 256])


        '''
        out0:   ([2, 256, 64, 64])
        out1:   ([2, 128, 128, 128])
        out2:   ([2, 64, 256, 256])
        '''
        return [out0, out1, out2]
    
    def forward(self, event_feature, frame_feature, warped_feature):
        ### forward encoder 
        '''
        event_feature   len: 3   0:([2, 32, 256, 256])   1:([2, 64, 128, 128])    2:([2, 128, 64, 64])
        frame_feature   len: 3   0:([2, 16, 256, 256])   1:([2, 32, 128, 128])    2:([2, 64, 64, 64])
        warped_feature  len: 3   0:([2, 22, 256, 256])   1:([2, 38, 128, 128])    2:([2, 70, 64, 64])
        '''
        # 只用到了 event_feature[0], frame_feature[0], warped_feature[0]
        Q_list = self.build_qurey(event_feature, frame_feature, warped_feature) # Q_list: len: 3   0:([2, 64, 256, 256])   1:([2, 128, 128, 128])    2:([2, 256, 64, 64])
        ### forward decoder
        FORWARD_B3=None
        out_decoder = self.forward_decoder(Q_list, warped_feature, frame_feature, event_feature)
        ### synthesis frame
        img0 = self.last_conv0(out_decoder[0]) # out_decoder[0]: ([2, 256, 64, 64])   -> img0: ([2, 3, 64, 64])
        img1 = self.last_conv1(out_decoder[1]) # out_decoder[1]: ([2, 128, 128, 128]) -> img1: ([2, 3, 128, 128])
        img2 = self.last_conv2(out_decoder[2]) # out_decoder[2]: ([2, 64, 256, 256])  -> img2: ([2, 3, 256, 256])
        '''
        Q_list      len: 3 ([2, 64, 256, 256]) ...
        '''
        #! return: 2->0 再次反过来了， 反反得正
        return [img2, img1, img0]

class EventInterpNet(nn.Module):
    def __init__(self, num_bins=16,  flow_debug=False):
        super(EventInterpNet, self).__init__()
        unit_dim = 32  #!
        # scale 
        self.scale = 3
        # flownet
        self.flownet = FlowNet(md=4, tb_debug=flow_debug)
        self.flow_debug = flow_debug
        # encoder
        self.encoder_f = frame_encoder(3, unit_dim//4)
        self.encoder_e = event_encoder(16, unit_dim//2)
        # decoder
        self.transformer = Transformer(unit_dim*2)
        # channel scaling convolution
        self.conv_list = nn.ModuleList([conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim)])

        self.feat_imnet_event_s1  = Siren(in_features=32+2, out_features=32, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        self.feat_imnet_frame_s1  = Siren(in_features=22+2, out_features=16, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        self.feat_imnet_warped_s1 = Siren(in_features=22+2, out_features=22, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        
        # self.feat_imnet_frame_s2 = Siren(in_features=38, out_features=38, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        # self.feat_imnet_frame_s3 = Siren(in_features=70, out_features=70, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)



    def bwarp(self, x, flo):
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
        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)
        return output * mask

    def Flow_pyramid(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(F.interpolate(flow, scale_factor=0.5 ** i, mode='bilinear') * (0.5 ** i))
        return flow_pyr

    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr
    
    # def feature_upsample(self, frame_feature, spatial_upscale=4):
    #     '''
    #     frame_feature len: 3 ([1, 16, 256, 384])
    #     '''
    #     frame_feature_ot = []
    #     for idx in range(self.scale):
    #         bs,cs,h,w = frame_feature[idx].shape
    #         hh,ww = int(h*spatial_upscale), int(w*spatial_upscale)
    #         coord_highres = make_coord((hh, ww)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda() # ([bs, hh*ww, 2])


    #         tmp = F.grid_sample(  # tmp ([1, 16, hh*ww])
    #                     frame_feature[idx],          # ([1, 16, 256, 384])                                                   # feat ([24, 192, 32, 32])  
    #                     coord_highres.flip(-1).unsqueeze(1),
    #                     mode='bilinear', align_corners=False)[:, :, 0, :].view(bs, cs, hh, ww)
    #         # g(tmp)
            
    #         # [:, :, 0, :]
    #         frame_feature_ot.append(tmp) 

    #     return frame_feature_ot

    def feature_upsample_sgl(self, frame_feature_sgl, spatial_upscale=None): # 4
        '''
        frame_feature_sgl : ([1, 16, 256, 384])
        '''
        # frame_feature_ot = []
        # for idx in range(self.scale):
        bs,cs,h_s,w_s = frame_feature_sgl.shape  # h_s 不同于 h, 与 for idx in range(self.scale) 有关
        hh_s,ww_s = int(h_s*spatial_upscale), int(w_s*spatial_upscale)
        
        coord_highres_s = make_coord((hh_s, ww_s), flatten=True).repeat(self.bs, 1, 1)\
                                                .clamp(-1 + 1e-6, 1 - 1e-6).cuda() 

        tmp = F.grid_sample(  # tmp ([1, 16, hh*ww])
                    frame_feature_sgl,          # ([1, 16, 256, 384])                                                   # feat ([24, 192, 32, 32])  
                    coord_highres_s.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :].view(self.bs, cs, hh_s, ww_s)
        # g(tmp)
        return tmp


    def feature_upsample_composite(self, frame_feature_, upsample_net ,spatial_upscale=None): #4
        '''
        in: 
            frame_feature_  :List, len 3  0:([2, 22, h, w])   1:([2, 38, h/2, w/2])    2:([2,  70, h/4, w/4])
        out:
            frame_feature   :List, len 3  0:([2, 22, hh, w])   1:([2, 38, hh/2, ww/2])    2:([2,  70, hh/4, ww/4])
        '''
        frame_feature=[]
        for idx in range(self.scale):
            if idx ==0:

                feat = frame_feature_[0]   # ([2, 22, h, w])

                q_feat = F.grid_sample(    # q_feat ([24, 16384, 192])
                    feat,                                                               # feat ([bs, 22, h, w])  
                    self.coord_highres.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  
 
                q_coord = F.grid_sample(   # q_coord ([24, 16384, 2])
                    self.feat_coord,                                                         # feat_coord ([24, 2, 32, 32])
                    self.coord_highres.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  
                rel_coord = self.coord_highres - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                inp = torch.cat([q_feat,  rel_coord], dim=-1)


                HRfeat = upsample_net( inp.view(self.bs * self.qs, -1) ).view(self.bs, self.qs, -1)
                HRfeat = HRfeat.permute(0, 2, 1).view(self.bs, -1, self.hh, self.ww)                       # HRfeat ([bs, -1, hh, ww])
                tmp = HRfeat


            else:
                tmp = self.feature_upsample_sgl(frame_feature_[idx], spatial_upscale)

            frame_feature.append(tmp)
        return frame_feature


    def synthesis(self, batch, OF_t0, OF_t1, spatial_upscale):   
        '''
        batch['image_input0']: ([bs, 3, h, w])
        batch['image_input1']: ([bs, 3, h, w])
        OF_t0 len:1  ([bs, 2, 256, 384])
        OF_t1 len:1  ([bs, 2, 256, 384])
        '''

        ## frame encoding
        f_frame0 = self.encoder_f(batch['image_input0'])        # f_frame0:List        len(a_list):3 ; [0]:([bs, 8, h, w]), [1]:([2, 16, 32, 32]), [2]:([2, 32, 16, 16])
        f_frame1 = self.encoder_f(batch['image_input1'])
        
        #* add from videoinr
        self.bs, _ , self.h, self.w = f_frame0[0].shape
        self.hh, self.ww = int(self.h * spatial_upscale), int(self.w * spatial_upscale)
        self.coord_highres = make_coord((self.hh, self.ww), flatten=True).repeat(self.bs, 1, 1)\
                                                .clamp(-1 + 1e-6, 1 - 1e-6).cuda()                  # coord_highres ([bs, hh*ww, 2])
        self.qs = self.coord_highres.shape[1]
        self.feat_coord    = make_coord((self.h,   self.w), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0)\
                                                .expand(self.bs, 2, self.h,   self.w)                # feat_coord   ([bs, 2, h, w])

        ## OF pyramid
        OF_t0_pyramid = self.Flow_pyramid(OF_t0[0])             # OF_t0_pyramid:List   len(a_list):3 ; [0]:([bs, 2, h, w]), [1]:([1, 2, 128, 192]), [2]:([1, 2, 64, 96])
        OF_t1_pyramid = self.Flow_pyramid(OF_t1[0])
        ## image pyramid
        I0_pyramid = self.Img_pyramid(batch['image_input0'])    # I0_pyramid:List       len(a_list):3 ; [0]:([bs, 3, h, w]), [1]:
        I1_pyramid = self.Img_pyramid(batch['image_input1'])
        # frame0_warped, frame1_warped = [], []


        warped_feature, frame_feature_ = [], []
        for idx in range(self.scale):
            frame0_warped = self.bwarp(torch.cat((f_frame0[idx], I0_pyramid[idx]),dim=1), OF_t0_pyramid[idx])  # channel 8+3=11
            frame1_warped = self.bwarp(torch.cat((f_frame1[idx], I1_pyramid[idx]),dim=1), OF_t1_pyramid[idx])
            warped_feature.append(torch.cat((frame0_warped, frame1_warped), dim=1))                            # channel 11*2 =22
            
            frame0 = torch.cat((f_frame0[idx], I0_pyramid[idx]),dim=1)    # channel 8+3=11
            frame1 = torch.cat((f_frame1[idx], I1_pyramid[idx]),dim=1)
            # frame_feature.append(torch.cat((f_frame0[idx], f_frame1[idx]), dim=1)) # frame_feature:  List, len: 3   0:([bs, 16, h, w]) 包含 image_input0  和 image_input1
            frame_feature_.append(torch.cat((frame0, frame1), dim=1))                # frame_feature_: List, len: 3   0:([bs, 22, h, w]) 包含 image_input0  和 image_input1
            # aaa = I0_pyramid[idx]
            # bbb = self.bwarp(aaa, OF_t0_pyramid[idx])
            # ts.save(aaa,'aaa.png') 
            # ts.save(bbb,'bbb.png')  
            # ts.save(OF_t0_pyramid[idx],'ccc.png')
        event_feature = []
        fr_feature = []
        # event encoding for frame interpolation
        f_event_0t = self.encoder_e(batch['event_input_0t'])
        f_event_t1 = self.encoder_e(batch['event_input_t1'])
        for idx in range(self.scale):
            event_feature.append(torch.cat((f_event_0t[idx], f_event_t1[idx]), dim=1))
            fr_feature.append(torch.cat((I0_pyramid[idx], I1_pyramid[idx]), dim=1))  # fr_feature:List    len(a_list):3;
        '''
        before feature_upsample:
            event_feature   :List, len 3  0:([2, 32, h, w])   1:([2, 64, h/2, w/2])    2:([2, 128, h/4, w/4])
            frame_feature_  :List, len 3  0:([2, 22, h, w])   1:([2, 38, h/2, w/2])    2:([2,  70, h/4, w/4])  22=16+6, 40=32+6, 70=64+6  #! channel=22
            warped_feature  :List, len 3  0:([2, 22, h, w])   1:([2, 38, h/2, w/2])    2:([2,  70, h/4, w/4])
            fr_feature      :List, len 3  0:([2,  6, h, w])   1:([2,  6, h/2, w/2])    2:([2,   6, h/4, w/4])   h=64, w=64 #* new
        '''

        # '''
        # after feature_upsample:
        #     event_feature   :List, len 3  0:([2, 32, hh, ww])   1:([2, 64, hh/2, ww/2])    2:([2, 128, hh/4, ww/4])
        #     frame_feature   :List, len 3  0:([2, 16, hh, ww])   1:([2, 32, hh/2, ww/2])    2:([2, 64, hh/4, ww/4])
        #     warped_feature  :List, len 3  0:([2, 22, hh, ww])   1:([2, 38, hh/2, ww/2])    2:([2, 70, hh/4, ww/4])
        #     fr_feature      :List, len 3  0:([2,  6, hh, ww])   1:([2,  6, hh/2, ww/2])    2:([2,  6, hh/4, ww/4])   h=64, w=64 #* new
        # after feature_upsample, event_feature  len: 3   0:([2, 32, 256, 256])   1:([2, 64, 128, 128])    2:([2, 128, 64, 64])
        # after feature_upsample, frame_feature  len: 3   0:([2, 16, 256, 256])   1:([2, 32, 128, 128])    2:([2, 64, 64, 64])
        # after feature_upsample, warped_feature len: 3   0:([2, 22, 256, 256])   1:([2, 38, 128, 128])    2:([2, 70, 64, 64])
        # '''
        # frame_feature  = self.feature_upsample(frame_feature, spatial_upscale=spatial_upscale)
        # warped_feature = self.feature_upsample(warped_feature, spatial_upscale=spatial_upscale)
        # event_feature  = self.feature_upsample(event_feature, spatial_upscale=spatial_upscale)
        
        flag = 'original'
        # flag = 'remove_upsample_net'
        if flag == 'original':
            # channel 32 ->32
            event_feature  = self.feature_upsample_composite(event_feature,  upsample_net = self.feat_imnet_event_s1, spatial_upscale=spatial_upscale)       
            # channel 16+6=22  -> 16
            frame_feature  = self.feature_upsample_composite(frame_feature_, upsample_net = self.feat_imnet_frame_s1, spatial_upscale=spatial_upscale)   
            frame_feature[1] = frame_feature[1][:, :-6, :, :]
            frame_feature[2] = frame_feature[2][:, :-6, :, :]  # todo 不优雅
            
            # chennel 22 ->22
            warped_feature = self.feature_upsample_composite(warped_feature, upsample_net = self.feat_imnet_warped_s1, spatial_upscale=spatial_upscale)   
            # warped_feature = [torch.zeros_like(warped_feature[i]) for i in range(self.scale)]
        elif flag == 'remove_upsample_net':
            event_feature = event_feature
            
            frame_feature = frame_feature_
            # frame_feature  = self.feature_upsample_composite(frame_feature_, upsample_net = self.feat_imnet_frame_s1, spatial_upscale=spatial_upscale)   
            frame_feature[0] = frame_feature[0][:, :-6, :, :]
            frame_feature[1] = frame_feature[1][:, :-6, :, :]
            frame_feature[2] = frame_feature[2][:, :-6, :, :]  # todo 不优雅
            
            warped_feature = warped_feature
        FORWARD_B2=None
        TRANSFORMER=None
        '''
        int:
            event_feature   :List, len 3  0:([2, 32, hh, ww])   1:([2, 64, hh/2, ww/2])    2:([2, 128, hh/4, ww/4])
            frame_feature   :List, len 3  0:([2, 16, hh, ww])   1:([2, 32, hh/2, ww/2])    2:([2, 64, hh/4, ww/4])   #! channel=16
            warped_feature  :List, len 3  0:([2, 22, hh, ww])   1:([2, 38, hh/2, ww/2])    2:([2, 70, hh/4, ww/4])
        out:
            img_out len:3  0:([2, 3, 256, 256])   1:([2, 3, 128, 128])    2:([2, 3, 64, 64])
        '''
        # g(warped_feature)
        img_out = self.transformer(event_feature, frame_feature, warped_feature) 
        output_clean = [] # todo  resolution
        for i in range(self.scale):  
            '''
            #output_clean len:3  0:([2, 3, 256, 256])   1:([2, 3, 128, 128])    2:([2, 3, 64, 64])
            '''
            output_clean.append(torch.clamp(img_out[i], 0, 1))
        return output_clean


    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy

    def cal_grad2_error(self, flow, img):
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
        loss_list = []
        for idx in range(self.scale):
            flow, img = optical_flows[idx], img_pyramid[idx]
            #error = self.cal_grad2_error(flow, img)
            error = self.cal_grad2_error(flow/20.0, img)
            loss_list.append(error[:,None])
        loss_flow_smooth = torch.cat(loss_list, 1).sum(1)
        return loss_flow_smooth
    

    def cal_flow_loss_after_flow(self, batch, OF_t0, OF_t1):
        ## OF pyramid
        OF_t0_pyramid = self.Flow_pyramid(OF_t0[0])             # OF_t0_pyramid:List   len(a_list):3 ; [0]:([bs, 2, h, w]), [1]:([1, 2, 128, 192]), [2]:([1, 2, 64, 96])
        OF_t1_pyramid = self.Flow_pyramid(OF_t1[0])
        ## image pyramid
        I0_pyramid = self.Img_pyramid(batch['image_input0'])    # I0_pyramid:List       len(a_list):3 ; [0]:([bs, 3, h, w]), [1]:
        I1_pyramid = self.Img_pyramid(batch['image_input1'])
        # frame0_warped, frame1_warped = [], []

        loss_flow_smooth = self.compute_loss_flow_smooth(OF_t0_pyramid, I0_pyramid)  + \
                            self.compute_loss_flow_smooth(OF_t1_pyramid, I1_pyramid)

        imgt_0bwarped = self.bwarp(   I0_pyramid[0], OF_t0[0])
        imgt_1bwarped = self.bwarp(   I1_pyramid[0], OF_t1[0])

        return  imgt_0bwarped,imgt_1bwarped,loss_flow_smooth

    def forward(self, batch, spatial_upscale, mode:str):
        """
        batch: Dict:
            image_input0_org: ([bs, 3, 240, 360])  # 
            image_input1_org: ([bs, 3, 240, 360])
            image_input0:       ([bs,  3, h, w])
            image_input1:       ([bs,  3, h, w])
            imaget_input:       ([bs,  3, h, w])
            event_input_t0:     ([bs, 16, h, w])
            event_input_0t:     ([bs, 16, h, w])
            event_input_t1:     ([bs, 16, h, w])
        """
        if mode=='flow':

            OF_t0, OF_t1 = self.flownet(batch)

            imgt_0bwarped,imgt_1bwarped,loss_flow_smooth = self.cal_flow_loss_after_flow( batch, OF_t0, OF_t1)
            output_clean_pyramid_1 = torch.zeros_like(imgt_0bwarped)

            return output_clean_pyramid_1, OF_t0[0], OF_t1[0],imgt_0bwarped,imgt_1bwarped,loss_flow_smooth
        
        elif mode=='joint':
            FORWARD_A1=None
            OF_t0, OF_t1 = self.flownet(batch)   # OF_t0 len:1  ([bs, 2, 64, 64])
            '''
            OF_t0 len:1  ([bs, 2, 256, 384])
            OF_t1 len:1  ([bs, 2, 256, 384])
            '''
            
            #!
            FORWARD_B1=None
            output_clean = self.synthesis(batch, OF_t0, OF_t1, spatial_upscale)
            '''
            #output_clean len:3  0:([bs, 3, 256, 256])   1:([bs, 3, 128, 128])    2:([bs, 3, 64, 64])
            '''

            imgt_0bwarped = torch.zeros_like(output_clean[0])
            imgt_1bwarped = torch.zeros_like(output_clean[0])
            loss_flow_smooth = 0

            return output_clean[0], OF_t0[0], OF_t1[0], imgt_0bwarped, imgt_1bwarped, loss_flow_smooth
        

