'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os
import sys
import random
import pickle

import numpy as np
import cv2
try:
    import lmdb  # optional: only used for lmdb-format datasets
except ImportError:
    lmdb = None
import torch
import torch.utils.data as data
import dataset.util as util
try:
    import mc  # import memcached
except ImportError:
    pass
from pdb import set_trace as bp
import glob

from os.path import join,basename,dirname, abspath   #,abspath,expanduser
try:
    sys.path.append(dirname(dirname(os.path.abspath(__file__))))
    from dataset.util import imresize_np

    from utils.utils_y import evresize_np
except ImportError:
    pass

from utils.utils_y import print_list,print_list_all, load_ev_npz_gopro,load_ev_npz_erf,load_ev_npz_bsergb_evreal,ev_n4_l_to_events_frames,ev_downsample
# from utils.events_to_frame import event_stream_to_frames
# from utils.utils_y import event_stream_to_voxel_frames
from global_config import dprint

import torch.nn.functional as F

import logging
logger = logging.getLogger('inrlog')
import time

# from dmfq.dmfq import get_info as g

# from dmfq.dmfq import get_file_path_list_intname

# LOADEVENT=True
LOADFLOW=False
# LOADEVENT=False


def get_file_path_list_intname(root, suffix='.png', verbose=False):
    '''
    root='/home/hjy/data/Adobe240/frame/train/720p_240fps_1'
    suffix='.png'
    '''
    files_path_l = [p for p in sorted(glob.glob(f'{os.path.expanduser(root)}/*{suffix}')) if os.path.isfile(p)]
    files_path_l.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if verbose:
        # print(f"==>> imgs_path_list: {files_path_l}")
        print_list_all(files_path_l)
    return files_path_l

class adobeDataset_ev_from_raw(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(adobeDataset_ev_from_raw, self).__init__()
        self.opt = opt

        self.spatial_upscale = opt['spatial_upscale']

        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        # self.half_N_frames = opt['N_frames'] // 2       # 7//2 = 3
        # self.LR_N_frames = 1 + self.half_N_frames       # 4
        # assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        
        self.num_midpoints = opt['N_frames']         # 7
        self.num_allpoints = self.num_midpoints + 2     #9

        
        # event option
        self.moments = 32
        self.h_ori = opt['h_ori']  
        self.w_ori = opt['w_ori']  
        # self.low_resolution = (625 // scale, 970 // scale)
        self.positive = 1
        self.negative = 0

        self.crop_x = opt['crop_x']  # 960
        self.crop_y = opt['crop_y'] # 1440
        self.h_new = opt['h_new']  # 960
        self.w_new = opt['w_new'] # 1440

        self.GT_root, self.LQ_root = os.path.expanduser(opt['dataroot_GT']), os.path.expanduser(opt['dataroot_LQ'])
        print(f"==>> self.GT_root: {self.GT_root}")
        print(f"==>> self.LQ_root: {self.LQ_root}")
        # self.data_type = self.opt['data_type']
        # self.LR_input = True if opt['GT_size'] != opt['LQ_size'] else False  # low resolution inputs

        # if self.data_type == 'lmdb':
        #     self.GT_env, self.LQ_env = None, None
        # elif self.data_type == 'mc':  # memcached
        #     self.mclient = None
        # elif self.data_type == 'img':
        #     pass
        # else:
        #     raise ValueError('Wrong data type: {}'.format(self.data_type))
        
        if self.opt['use_one_sce_for_train_debug'] == True:
            with open('dataset/erf_folder_train_one_sce_for_debug.txt') as t:
                video_list = t.readlines()
        else:
            with open(opt['video_list_txt']) as t:
                video_list = [line.rstrip('\n') for line in t.readlines()]
                # video_list = video_list[:50]
            
        if self.opt.get('is_test', False): 
            try:
                video_list.remove('0024') 
            except:
                pass

        self.file_list = []
        self.gt_list = []

        self.ev_npz_llist = []
        print_list(video_list)
        self.video_list = video_list


        # hq_folder_name = '1_2_images'  # '1_2_images' 480
        # hq_folder_name = 'processed_images'
        # lq_folder_name = 'processed_images'
        # ev_folder_name = 'processed_events'
        hq_folder_name = ''
        lq_folder_name = ''
        ev_folder_name = ''



        if opt['ev_sim']== 'volt':
            if opt['is_train']:
                ev_voxel_root_new = join(self.GT_root,'../train_volt') # join(dirname(self.GT_root),'train_mini')# ../data/ERF/train ——> ../data/ERF/train/train_ev_voxel
            elif opt['is_test']:
                ev_voxel_root_new = join(self.GT_root,'../test_volt') #join(dirname(self.GT_root),'test_mini')# ../data/ERF/train ——> ../data/ERF/train/train_ev_voxel
        elif opt['ev_sim']== 'vid2e':
            if opt['is_train']:
                ev_voxel_root_new = join(self.GT_root,'../train_vid2e') # join(dirname(self.GT_root),'train_mini')# ../data/ERF/train ——> ../data/ERF/train/train_ev_voxel
            elif opt['is_test']:
                ev_voxel_root_new = join(self.GT_root,'../test_vid2e') #join(dirname(self.GT_root),'test_mini')# ../data/ERF/train ——> ../data/ERF/train/train_ev_voxel

        for video in video_list:
            if video[-1] == '\n':
                video = video[:-1]
            index = 0
            # frames = (os.listdir(join(self.GT_root , video, hq_folder_name)))
            # frames = sorted([frame for frame in frames if frame.endswith('.png') ])
            frames = get_file_path_list_intname(root=join(self.GT_root , video, hq_folder_name), suffix='.png', verbose=False) #! 报过bug ,adobe 这里要注意改的方式，直接abspath是有错误的
            frames = [os.path.basename(f) for f in frames] #  1.png


            # frames_LQ = (os.listdir(join(self.GT_root , video, lq_folder_name)))
            # frames_LQ = sorted([frame for frame in frames_LQ if frame.endswith('.png') ])
            frames_LQ = get_file_path_list_intname(root=join(self.GT_root , video, lq_folder_name), suffix='.png', verbose=False)
            frames_LQ = [os.path.basename(f) for f in frames_LQ] #  1.png
            

            if self.opt.get('is_test', False):  # dict get 防止键不存在  #opt['is_test']==True:  # todo 加到别的 dataset 里面
                num_imgs = self.opt['test_num_imgs'] 
                # 48  # 49  # 00000.png ~ 00048.png
            elif self.opt.get('is_train', False):   
                if self.opt.get('train_all', False): 
                    num_imgs = len(frames)
                else:
                    num_imgs = self.opt['train_num_imgs']  # 100
            
            num_imgs = min(num_imgs,len(frames)) #* add

            logger.info(f"==>> num_imgs: {num_imgs}")
            frames = frames[:num_imgs]
            # print_list_all(frames)
            frames_LQ = frames_LQ[:num_imgs]
            logger.debug(f"==>> frames_LQ: {frames_LQ}")

            ev_root =  abspath(join(ev_voxel_root_new , video, ev_folder_name))  # video='0000
            # print(f"==>> ev_root: {ev_root}")
            # npz_list = sorted(glob.glob(f'{ev_root}/*.npz'))
            npz_list = get_file_path_list_intname(root=ev_root, suffix='.npz', verbose=False)


            # todo to remove
            # st_idx =1200
            # ed_idx =1219
            # st_idx =1370
            # ed_idx =1540
            # frames = frames[st_idx:ed_idx]  #[968:1048]
            # frames_LQ = frames_LQ[st_idx:ed_idx]
            # num_imgs = len(frames)
            # print(f"==>> frames: {frames}")

            sce_file_list = []
            sce_gt_list = []
            sce_ev_npz_llist = []
            #* 1/10 mini trainset
            while index + self.num_midpoints * 1 + 1 <= num_imgs-1:   #  99 0~91
                videoInputs  = [frames_LQ[i] for i in      [index, index + 1 + self.num_midpoints] ]
                video_all_gt = [frames[i] for i in range(index, index + 2 + self.num_midpoints)]
                videoInputs = [join(video, hq_folder_name, f) for f in videoInputs]
                videoGts    = [join(video, hq_folder_name, f) for f in video_all_gt]
                sce_file_list.append(videoInputs)
                sce_gt_list.append(videoGts)

                # tmp1= [f'{ev_root}/{index+i:0>6d}.npz'  for i in range(1,self.num_allpoints)] #! bug 的源头，错误的是 for i in range(self.num_allpoints)
                tmp1= [npz_list[index+i] for i in range(1,self.num_allpoints)] #! bug 的源头，错误的是 for i in range(self.num_allpoints)
                
                sce_ev_npz_llist.append(tmp1)
                index += 1

            self.file_list.append(sce_file_list)
            self.gt_list.append(sce_gt_list)
            self.ev_npz_llist.append(sce_ev_npz_llist)


        if self.opt.get('is_test', False):
            self.file_list = [item for sublist in self.file_list for item in sublist[::(self.num_midpoints+1)]]    # 8
            self.gt_list = [item for sublist in self.gt_list for item in sublist[::(self.num_midpoints+1)]]
            self.ev_npz_llist = [item for sublist in self.ev_npz_llist for item in sublist[::(self.num_midpoints+1)]]
        else:
            self.file_list = [item for sublist in self.file_list for item in sublist]
            self.gt_list = [item for sublist in self.gt_list for item in sublist]
            self.ev_npz_llist = [item for sublist in self.ev_npz_llist for item in sublist]


        print(f'len(self.file_list):',len(self.file_list))
        print_list(self.file_list)
        
        print(f'len(self.gt_list):',len(self.gt_list))
        print_list(self.gt_list)

        print(f'len(self.ev_npz_llist):',len(self.ev_npz_llist))
        print_list(self.ev_npz_llist)

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img


    def _load_img(self,index,gt_sampled_idx_l,lq_scale,hq_scale):
        img_lq_path_l = [join(self.LQ_root, fp) for fp in self.file_list[index]]
        img_gt_path_l = [join(self.GT_root, fp) for fp in self.gt_list[index]]
        img_gt_path_l = [img_gt_path_l[i] for i in gt_sampled_idx_l]        

        logger.debug(f"==>> img_lq_path_l: {img_lq_path_l}")
        logger.debug(f"==>> img_gt_path_l: {img_gt_path_l}")

        img_lq_l = [(cv2.imread(fp)[self.crop_x:self.crop_x+self.h_new,self.crop_y:self.crop_y+self.w_new,:]/255.) for fp in img_lq_path_l] # List[np.ndarray] 2  (624, 960, 3)
        img_gt_l = [(cv2.imread(fp)[self.crop_x:self.crop_x+self.h_new,self.crop_y:self.crop_y+self.w_new,:]/255.) for fp in img_gt_path_l] # List[np.ndarray] 9  (624, 960, 3)

        img_lq_l = [imresize_np(fp, lq_scale , True) for fp in img_lq_l]
        img_gt_l = [imresize_np(fp, hq_scale , True) for fp in img_gt_l]
        
        IMG_NORM=None
        img_lq_l = [ (np.clip(fp,0,1)*255).astype(np.uint8)    for fp in img_lq_l]
        img_gt_l = [ (np.clip(fp,0,1)*255).astype(np.uint8)    for fp in img_gt_l]
        
        
        img_lq_l = [img_.astype(np.float32) / 255. for img_ in img_lq_l]
        img_gt_l = [img_.astype(np.float32) / 255. for img_ in img_gt_l]
            
        if img_lq_l[0].ndim == 2:
            img_lq_l = [np.expand_dims(img_, axis=2) for img_ in img_lq_l]
            img_gt_l = [np.expand_dims(img_, axis=2) for img_ in img_gt_l]
            
        if img_lq_l[0].shape[2] > 3:
            img_lq_l = [img_[:, :, :3] for img_ in img_lq_l]
            img_gt_l = [img_[:, :, :3] for img_ in img_gt_l]
                
        return img_lq_l,img_gt_l




    def _load_event(self,ev_n4_l,gt_sampled_idx,h_ori,w_ori,lq_scale):
        #* func
        ev_0t_n4_l = ev_n4_l[0:gt_sampled_idx]      # [0:0] -> 0个     [0:8] 8个   1,2,...,8  
        ev_t1_n4_l = ev_n4_l[gt_sampled_idx:]       # [0:]  -> 8个     [8:]  0个
        ev_0t_voxel_hr = ev_n4_l_to_events_frames(ev_0t_n4_l,num_bins=16,height=h_ori,width=w_ori)          # 16,975,1440
        ev_t1_voxel_hr = ev_n4_l_to_events_frames(ev_t1_n4_l,num_bins=16,height=h_ori,width=w_ori)          # 16,975,1440

        #* crop to 960,1440
        ev_0t_voxel_hr = ev_0t_voxel_hr[:,self.crop_x:self.crop_x+self.h_new,self.crop_y:self.crop_y+self.w_new]
        ev_t1_voxel_hr = ev_t1_voxel_hr[:,self.crop_x:self.crop_x+self.h_new,self.crop_y:self.crop_y+self.w_new]



        lr_h,lr_w =int(self.h_new*lq_scale), int(self.w_new*lq_scale)   # 120,180    240,360


        ev_0t_voxel = (ev_downsample(torch.from_numpy(ev_0t_voxel_hr[None,:,:,:]),lr_h=lr_h ,lr_w=lr_w ))[0,:,:,:].numpy().astype(np.float16) # 16,120,180
        ev_t1_voxel = (ev_downsample(torch.from_numpy(ev_t1_voxel_hr[None,:,:,:]),lr_h=lr_h ,lr_w=lr_w ))[0,:,:,:].numpy().astype(np.float16) # 16,120,180


        return ev_0t_voxel,ev_t1_voxel
    
    
    def __getitem__(self, index):

        
        if self.opt['is_train']: #   : true
            self.spatial_upscale = self.spatial_upscale #random.randint(1, 4)  
            self.hq_scale = 1/2
            self.lq_scale = self.hq_scale * (1 /self.spatial_upscale)
        elif self.opt['is_test']:
            self.spatial_upscale = self.opt['spatial_upscale']
            self.hq_scale = 1
            self.lq_scale = self.hq_scale * (1 /self.spatial_upscale)
        logger.info(f"==>> self.spatial_upscale: {self.spatial_upscale}")



        logger.debug(f"==>> index: {index}")

        start_time = time.time()
        h_ori = self.opt['h_ori'] # 975
        w_ori = self.opt['w_ori'] # 1440

        h_new = self.opt['h_new']  # 960
        w_new = self.opt['w_new']  # 1440


        N_frames = self.opt['N_frames']
        if self.opt['phase'] == 'train':
            GT_size = self.opt['GT_size']
            LQ_size = int(self.opt['GT_size']/self.opt['spatial_upscale'])
        else:
            GT_size = None
            LQ_size = None

        # center_frame_idx = random.randint(2,6) # 2<= index <=6

        #### determine the neighbor frames
        # intervl = random.choice(self.interval_list)
        # if self.opt['border_mode']:
        #     direction = 1  # 1: forward; 0: backward
        #     if self.random_reverse and random.random() < 0.5:
        #         direction = random.choice([0, 1])
        #     if center_frame_idx + intervl * (N_frames - 1) > 7:
        #         direction = 0
        #     elif center_frame_idx - intervl * (N_frames - 1) < 1:
        #         direction = 1
        #     # get the neighbor list
        #     if direction == 1:
        #         neighbor_list = list(
        #             range(center_frame_idx, center_frame_idx + intervl * N_frames, intervl))
        #     else:
        #         neighbor_list = list(
        #             range(center_frame_idx, center_frame_idx - intervl * N_frames, -intervl))
        # else:
        #     # ensure not exceeding the borders
        #     while (center_frame_idx + self.half_N_frames * intervl >
        #            7) or (center_frame_idx - self.half_N_frames * intervl < 1):
        #         center_frame_idx = random.randint(2, 6)
        #     # get the neighbor list
        #     neighbor_list = list(
        #         range(center_frame_idx - self.half_N_frames * intervl,
        #               center_frame_idx + self.half_N_frames * intervl + 1, intervl))
        #     if self.random_reverse and random.random() < 0.5:
        #         neighbor_list.reverse()

        #! SAMPLE_INDEX
        SAMPLE_INDEX=None
        if self.opt.get('is_train', False): 
            #* time random sample ,  one value each sample
            gt_sampled_idx_l = sorted(random.sample(range(self.num_allpoints), 1)) 
        elif self.opt.get('is_test', False):
            gt_sampled_idx_l = [i for i in range(self.num_allpoints)] # [0,1,2..8]        #### img path list
        logger.debug(f"==>> gt_sampled_idx: {gt_sampled_idx_l}")
        
        TIMES_L =None
        ##### times list    # [0.5]  len(times):1
        times_l = [torch.tensor([i / (self.num_allpoints-1)]) for i in gt_sampled_idx_l]


        #! IMGLOADING  cv2.imread
        IMGLOADING=None
        img_lq_l,img_gt_l = self._load_img(index,gt_sampled_idx_l,lq_scale=self.lq_scale,hq_scale=self.hq_scale)

        h,w,_ = img_lq_l[0].shape
        hh,ww,_ = img_gt_l[0].shape
    
        
        # #! EVENTLOADING  np.load
        if self.opt['load_event']:

            ev_npz_list = self.ev_npz_llist[index]
            ev_n4_l = [load_ev_npz_gopro(i) for i in ev_npz_list]

            EVENTLOADING=None
            ev_0t_voxel_lq_l=[]
            ev_t1_voxel_lq_l=[]
            ev_01_voxel_lq_l=[]
            _,ev_t1_voxel = self._load_event(ev_n4_l,gt_sampled_idx=0, h_ori=self.h_ori, w_ori=self.w_ori , lq_scale=self.lq_scale)
            ev_01_voxel_lq_l= [ev_t1_voxel for _ in range(len(gt_sampled_idx_l))]
            for gt_sampled_idx in gt_sampled_idx_l:
                # gt_sampled_idx = gt_sampled_idx_l[i] # 0 or 1  ... or 8
                try:
                    ev_0t_voxel,ev_t1_voxel = self._load_event(ev_n4_l,gt_sampled_idx, h_ori=self.h_ori, w_ori=self.w_ori , lq_scale=self.lq_scale)
                except Exception as e:
                    print("Error:", e)
                    print(f"==>> ev_npz_list: {ev_npz_list}")
                ev_0t_voxel_lq_l.append(ev_0t_voxel)
                ev_t1_voxel_lq_l.append(ev_t1_voxel)

        else:
            ev_0t_voxel_lq_l = [np.zeros((16,h,w)) for _ in range(len(gt_sampled_idx_l))]
            ev_t1_voxel_lq_l = [np.zeros((16,h,w)) for _ in range(len(gt_sampled_idx_l))]
            ev_01_voxel_lq_l = [np.zeros((16,h,w)) for _ in range(len(gt_sampled_idx_l))]


        ev_0t_voxel_lq_l = [np.transpose(v,(1,2,0)) for v in ev_0t_voxel_lq_l ]  # List[np.ndarray] len: 9 or 1,  (120, 180, 16) 
        ev_t1_voxel_lq_l = [np.transpose(v,(1,2,0)) for v in ev_t1_voxel_lq_l ]  # List[np.ndarray] len: 9 or 1,  (120, 180, 16) 
        ev_01_voxel_lq_l = [np.transpose(v,(1,2,0)) for v in ev_01_voxel_lq_l ]  # List[np.ndarray] len: 9 or 1,  (120, 180, 16) 


        tmp1 = np.zeros((hh,ww,2))
        tmp2 = np.zeros((hh,ww,2))
        flow_raft_l = [tmp1,tmp2]

        # todo  ev 没有normalize
        if self.opt['phase'] == 'train':
            # randomly crop
            CROP_RANDOM=None
            rnd_h = random.randint(0, max(0, h - LQ_size))
            rnd_w = random.randint(0, max(0, w - LQ_size))
            img_lq_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_lq_l]
            
            # evt_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in evt_LQ_l]
            ev_0t_voxel_lq_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in ev_0t_voxel_lq_l]
            ev_t1_voxel_lq_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in ev_t1_voxel_lq_l]
            ev_01_voxel_lq_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in ev_01_voxel_lq_l]

            rnd_h_HR, rnd_w_HR = int(rnd_h * self.spatial_upscale), int(rnd_w * self.spatial_upscale)
            img_gt_l = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_gt_l]

            flow_raft_l = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in flow_raft_l]


            #! FLIP_ROT
            FLIP_ROT=None
            # augmentation - flip, rotate
            data_dict={"img_lq_l":img_lq_l,
                       "img_gt_l":img_gt_l,
                       "ev_0t_voxel_lq_l":ev_0t_voxel_lq_l,
                       "ev_t1_voxel_lq_l":ev_t1_voxel_lq_l,
                       "ev_01_voxel_lq_l":ev_01_voxel_lq_l,
                       "flow_raft_l":flow_raft_l,
                       }
            data_dict = util.augment_dict(data_dict, self.opt['use_flip'], self.opt['use_rot'])

            img_lq_l = data_dict['img_lq_l']  
            img_gt_l = data_dict['img_gt_l']  
            ev_0t_voxel_lq_l = data_dict['ev_0t_voxel_lq_l'] 
            ev_t1_voxel_lq_l = data_dict['ev_t1_voxel_lq_l'] 
            ev_01_voxel_lq_l = data_dict['ev_01_voxel_lq_l'] 
            flow_raft_l = data_dict['flow_raft_l'] 


        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_lq_l, axis=0)        # hw3 -> 2hw3        (2, 120, 180, 3)
        img_flows = np.stack(flow_raft_l, axis=0)   # hw2 -> 2hw2        (2, 480, 720, 2)


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = np.transpose(img_LQs[:, :, :, [2, 1, 0]] , (0, 3, 1, 2))  # 2hw3 -> 2,3,h,w  BGR->RGB
        img_flows = np.transpose(img_flows, (0, 3, 1, 2))                   # 2hw2 -> 2,2,h,w  BGR->RGB


        img_GTs_l = [ np.transpose(f[None,:,:,[2, 1, 0]], (0, 3, 1, 2)) for f in img_gt_l]        # List len=9 , hh,ww,3  -> 1,hh,ww,3 -> 1,3,hh,ww  BGR->RGB
        ev_0t_voxel_lq_l = [ np.transpose(f[None,:,:,:], (0, 3, 1, 2)) for f in ev_0t_voxel_lq_l] # List len=9,  h, w,16  -> 1,h,w,16 -> 1,16,h,w        (120, 180, 16)   
        ev_t1_voxel_lq_l = [ np.transpose(f[None,:,:,:], (0, 3, 1, 2)) for f in ev_t1_voxel_lq_l]
        ev_01_voxel_lq_l = [ np.transpose(f[None,:,:,:], (0, 3, 1, 2)) for f in ev_01_voxel_lq_l]


        img_LQs          = torch.from_numpy(np.ascontiguousarray(img_LQs)).float()                             # ([2, 3, 32, 32])
        img_flows        = torch.from_numpy(np.ascontiguousarray(img_flows)).float()  # ([2, 2, 128, 128])
        img_GTs_l        = [torch.from_numpy(np.ascontiguousarray(f)).float() for f in img_GTs_l] # List len=9   ([1, 3, 480, 720])
        ev_0t_voxel_lq_l = [torch.from_numpy(np.ascontiguousarray(f)).float() for f in ev_0t_voxel_lq_l]        # ([1, 16, 32, 32])
        ev_t1_voxel_lq_l = [torch.from_numpy(np.ascontiguousarray(f)).float() for f in ev_t1_voxel_lq_l]        # ([1, 16, 32, 32])
        ev_01_voxel_lq_l = [torch.from_numpy(np.ascontiguousarray(f)).float() for f in ev_01_voxel_lq_l]        # ([1, 16, 32, 32])

        sce_name_list   = [ f.split('/')[-2]  for f in self.gt_list[index]]
        gt_png_name_list= [ f.split('/')[-1] for f in self.gt_list[index]]
        lq_png_name_list= [ f.split('/')[-1] for f in self.file_list[index]]

        end_time = time.time()

        # logger.info("#### erfDataset_ev __getitem__ 耗时: {:.2f}秒".format(end_time - start_time))
        logger.debug(f"==>> img_GTs.shape[-2], img_GTs.shape[-1]: {img_GTs_l[0].shape[-2], img_GTs_l[0].shape[-1]}")

        return {'LQs': img_LQs,                    # ([2, 3, 32, 32])
                'GT': img_GTs_l,                       #  List len=9, [0]: ([1, 3, 480, 720])    # todo   #  # [img_GTs],    #  list len 1
                'spatial_upscale': self.spatial_upscale, 
                                                                                                                  #! getitem 里面返回的如果是list,元素是 tensor ， 那个 batch 是 List[Tensor:增加0维度=bs]
                'sce_name_l':  sce_name_list,            #  List len=9, [0]: '0000'           [1]: '0000'...
                'gt_png_name_l':  gt_png_name_list,      #  List len=9, [0]: '00000.png',     [1]: '00001.png'...  #! getitem 里面返回的如果是list,元素是 str ， 那个 batch 是 List[List:len=bs]
                'lq_png_name_l':lq_png_name_list,
                'key': '00001_0001', 
                'time': times_l,      # list len 1
                'scale': (img_GTs_l[0].shape[-2], img_GTs_l[0].shape[-1]),  # 256 256
                'ev_0t_voxel_LQ_l':ev_0t_voxel_lq_l,  #  list len 1 or 9         bs,1,16,h,w
                'ev_t1_voxel_LQ_l':ev_t1_voxel_lq_l,  #  list len 1 or 9         bs,1,16,h,w
                'ev_01_voxel_LQ_l':ev_01_voxel_lq_l,  #  list len 1 or 9         bs,1,16,h,w
                'Flow':[img_flows]                    #  list len 1 or 9
                }


    def __len__(self):
        return len(self.file_list)


