
import numpy as np
import torch.nn.functional as F
import torch
from typing import List, Tuple

from .timers import Timer, CudaTimer

# import torchshow as ts

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import math
import socket
def get_host_name():
    hostname_str = socket.gethostname()
    return hostname_str


import pandas as pd



import os
from datetime import datetime
import pandas as pd


import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_color_map(error_hw,vmin=0.0, vmax=1.0): # vmin, vmax 有用
    '''
    y 20240712 经过测试，这种方法, 速度不受影响
    param:
        error_hw: h,w      0.-1. float32
    return:
        errmap_hw3_BGR255: h,w,3    0-255 uint8   BGR
    
    example:
        H, W = 50, 50
        error_hw = np.linspace(0.0, 1.0, H * W).reshape(H, W)  # 生成一个线性的错误矩阵]
        errmap_hw3_BGR255 = get_color_map(error_hw,vmin=0.0, vmax=1.0)
    '''
    
    assert vmax>vmin
    error_hw = np.clip(error_hw, vmin, vmax)  # 将错误矩阵限制在 [vmin, vmax] 区间
    error_hw = (error_hw-vmin)/(vmax-vmin)  # 将错误矩阵归一化到 [0, 1] 区间
    error_hw = np.clip(error_hw*255,0,255).astype(np.uint8)   # 生成一个线性的错误矩阵
    # print(f"==>> error_hw.shape: {error_hw.shape}")

    # 将灰度图像转换为Jet样式的伪彩色图像
    jet_image = cv2.applyColorMap(error_hw, cv2.COLORMAP_JET)

    errmap_hw3_BGR255 = jet_image
    # print(f"==>> errmap_hw3_BGR255.shape: {errmap_hw3_BGR255.shape}")
    # print(f"==>> errmap_hw3_BGR255.dtype: {errmap_hw3_BGR255.dtype}")
    return errmap_hw3_BGR255 # BGR


class CSVWriter_y:
    '''
    csv_writer = CSVWriter_y(file_path="output_without_header.csv")
    for i in range(6):
        new_data = [i, i*10, i*20, i*30]
        csv_writer.write(new_data)
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.current_row = 0
        self._initialize_csv()

    def _initialize_csv(self):
        CSVWriter_y.rename_if_file_exists(self.file_path)
        open(self.file_path, 'w').close()   # 创建空文件或清空现有文件
        self.current_row = 0                # 初始化时不包含header行

    def write(self, new_data):
        new_data_df = pd.DataFrame([new_data])          # 将新数据转换为 DataFrame
        new_data_df.to_csv(self.file_path, index=False, mode='a', header=False)  # 动态写入 CSV 文件
        self.current_row += len(new_data_df)            # 更新当前行数
        print(f"数据写入完成: self.file_path: {self.file_path}, new_data:",new_data)

    def get_row_count(self):
        return self.current_row

    @staticmethod
    def rename_if_file_exists(file_path):
        if os.path.isfile(file_path):                   # 如果存在， 且为文件路径
            _, suffix = os.path.splitext(file_path)     # eg.  '.csv'
            backup_file_path = f"{file_path.rsplit('.', 1)[0]}-archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"+ suffix
            os.rename(file_path, backup_file_path)
            print(f"文件已存在，重命名为: {backup_file_path}")
        else:
            print(f'文件不存在: {file_path}')



class ExcelWriter_y:
    '''
    powered by chatGPT and hjy
    example:
        # 定义第一行的说明字符串
        header = ["ID", "Value1", "Value2", "Value3"]
        out_excel_path = "output_with_header.xlsx"

        # 创建 ExcelWriter 实例
        excel_writer = ExcelWriter(out_excel_path, header)

        # 模拟循环体逐行获取数据并写入
        for i in range(6):  # 示例循环次数
            # 获取一行数据（这里用示例数据）
            new_data = [i, i*10, i*20, i*30]
            excel_writer.write(new_data)
    '''
    def __init__(self, file_path, header):
        self.file_path = file_path
        self.header = header
        self.current_row = 0
        self._initialize_excel()

    def _initialize_excel(self):
        # 初始化一个空的 DataFrame 并写入 header
        df = pd.DataFrame(columns=self.header)
        df.to_excel(self.file_path, index=False, engine='openpyxl')
        self.current_row = 1  # 初始化时包含header行

    def write(self, new_data):
        # 将新数据列表转换为字典
        new_data_dict = {self.header[i]: new_data[i] for i in range(len(new_data))}

        # 将新数据转换为 DataFrame
        new_data_df = pd.DataFrame([new_data_dict])
        # 动态写入 Excel 文件
        with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            # 获取当前 Sheet 的最大行数，用于确定新数据的起始写入位置
            writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
            startrow = self.current_row
            new_data_df.to_excel(writer, index=False, header=False, startrow=startrow)

        # 更新当前行数
        self.current_row += len(new_data_df)
        print("数据写入完成")

    def get_row_count(self):
        return self.current_row



class MetricTracker:
    '''
    powered by chatGPT
    '''
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if key not in self._data.index:
            self._data.loc[key] = [0, 0, 0]  # 添加新指标并初始化

        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.at[key, 'total'] += value * n
        self._data.at[key, 'counts'] += n
        self._data.at[key, 'average'] = self._data.at[key, 'total'] / self._data.at[key, 'counts']

    def avg(self, key):
        return self._data.at[key, 'average']
    
    def result(self):
        return dict(self._data['average'])

    def keys(self):
        return list(self._data.index)
    


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

def calc_lpips_pytorch_float(img1, img2, net='alex'):
    '''
    powered by chatGPT y:
    useage:     
        lpips_val = calc_lpips_pytorch_float(img1_tensor, img2_tensor)
    img1,img2: 
        1,3,H,W    0.0~1.0   RGB
    '''
    """
    Calculate the LPIPS distance between two images.

    Args:
        img1 (torch.Tensor): The first image tensor with shape (1, 3, H, W) and values in the range [0.0, 1.0].
        img2 (torch.Tensor): The second image tensor with shape (1, 3, H, W) and values in the range [0.0, 1.0].
        net (str): The backbone network for LPIPS. Options are 'alex', 'vgg', 'squeeze'. Default is 'alex'.

    Returns:
        torch.Tensor: The LPIPS distance.
    """

    import lpips  # lazy import: only needed when cal_lpips=true
    device = img1.device
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)

    # Ensure the inputs are torch tensors and have the correct shape
    assert isinstance(img1, torch.Tensor) and img1.shape[0] == 1 and img1.shape[1] == 3
    assert isinstance(img2, torch.Tensor) and img2.shape[0] == 1 and img2.shape[1] == 3

    # Ensure the input range is [0.0, 1.0]
    img1 = img1 * 2 - 1  # Normalize to [-1, 1] as required by LPIPS
    img2 = img2 * 2 - 1  # Normalize to [-1, 1] as required by LPIPS

    # Calculate the LPIPS distance
    lpips_val = loss_fn(img1, img2)

    return float(lpips_val)

def calc_psnr_pytorch(sr, hr):

    '''
    NCHW格式的tensor, uint8 0-255
    '''
    diff = (sr/255.0 - hr/255.0) 
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)


def calc_psnr_pytorch_float(sr, hr):

    '''
    NCHW格式的tensor, float 0.-1.
    '''
    diff = sr - hr
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)


def calc_ssim_pytorch(sr, hr):
    '''
    NCHW格式的tensor, uint8 0-255
    '''
    # def ssim(
    #     X,
    #     Y,
    #     data_range=255,
    #     size_average=True,
    #     win_size=11,
    #     win_sigma=1.5,
    #     win=None,
    #     K=(0.01, 0.03),
    #     nonnegative_ssim=False,
    # )
    ssim_val = ssim(sr/255.0, hr/255.0, data_range=1.0, size_average=True)
    return float(ssim_val)

def calc_ssim_pytorch_float(sr, hr):
    '''
    NCHW格式的tensor, float 0.- 1.
    '''
    # def ssim(
    #     X,
    #     Y,
    #     data_range=255,
    #     size_average=True,
    #     win_size=11,
    #     win_sigma=1.5,
    #     win=None,
    #     K=(0.01, 0.03),
    #     nonnegative_ssim=False,
    # )
    ssim_val = ssim(sr, hr, data_range=1.0, size_average=True)
    return float(ssim_val)

def logger_list(loger, a_list):
    import inspect
    frame = inspect.currentframe().f_back
    name_a_list=None
    for name, value in frame.f_locals.items():
        if value is a_list:
            name_a_list = name
    loger.info(' ')
    if name_a_list is not None:
        loger.info(f'    {name_a_list}, len: {len(a_list)}')
    for i, v in enumerate(a_list):
        if i<3 or i > len(a_list)-3:
            loger.info(f"    {i}: {v}")
        elif i==3:
            loger.info(f"    ...")

def logger_list_all(loger, a_list):
    import inspect
    frame = inspect.currentframe().f_back
    name_a_list=None
    for name, value in frame.f_locals.items():
        if value is a_list:
            name_a_list = name
    if name_a_list is not None:
        loger.info(f'    {name_a_list}')
    for i, v in enumerate(a_list):

        loger.info(f"        {i}: {v}")


def print_list(a_list):
    import inspect
    frame = inspect.currentframe().f_back
    name_a_list=None
    for name, value in frame.f_locals.items():
        if value is a_list:
            name_a_list = name
    print(' ')
    if name_a_list is not None:
        print(f'    {name_a_list}, len: {len(a_list)}')
    for i, v in enumerate(a_list):
        if i<3 or i > len(a_list)-3:
            print(f"    {i}: {v}")
        elif i==3:
            print(f"    ...")

def print_list_all(a_list):
    import inspect
    frame = inspect.currentframe().f_back
    name_a_list=None
    for name, value in frame.f_locals.items():
        if value is a_list:
            name_a_list = name
    if name_a_list is not None:
        print(f'    {name_a_list}')
    for i, v in enumerate(a_list):

        print(f"        {i}: {v}")


def get_save_idx_list(st_orser=3,ed_order=5):
    '''
    order = 0, return 1,2,3...9
    order = 1, return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    order = 2, return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    # # 调用函数生成数列
    # st_orser = 3
    # ed_order = 5
    # save_idx_list = get_save_idx_list(st_orser,ed_order)

    # for i in range(int(10**(ed_order+1))):
    #     if i in save_idx_list:
    #         print(i)
    ......
    '''
    count=0
    save_idx_list=[]
    for i in range(st_orser, ed_order+1):
        st = 10**i #
        ed = 9*10**i
        for j in range(st,ed+1,10**i):
            save_idx_list.append(j)
    # print(f"==>> len(save_idx_list): {len(save_idx_list)}")
    # print(f"==>> save_idx_list: {save_idx_list}")
    return save_idx_list



def get_numpy_info(varNp):
   
    '''
    import sys 
    sys.path.append("/Users/hjy/workspace/00_utils_sync/utils_folder")
    from utils import get_numpy_info
    '''
    # print("-"*20)
    print("".center(50, "-"))
    print(f"nummpy's shape is :{varNp.shape}")
    print(f"nummpy's dtype is :{varNp.dtype}")
    print(f"nummpy's max  num is :{varNp.max()}")
    print(f"nummpy's mean num is :{varNp.mean()}")
    print(f"nummpy's min  num is :{varNp.min()}")

    # print(f"==>> varNp: {varNp}")
    # plt.hist(arrayNp.ravel(), bins='auto')
    # plt.ylim(10000)
    # plt.hist(arrayNp.ravel(), bins=255)
    # plt.show()
    # plt.close()

def get_tensor_info(varTorch):
    '''
    import sys 
    sys.path.append("/Users/hjy/workspace/00_utils_sync/utils_folder")
    from utils import get_numpy_info
    '''
    # print("-"*20)
    print("".center(50, "-"))
    print(f"tensor's shape is :{varTorch.shape}")
    print(f"tensor's dtype is :{varTorch.dtype}")
    print(f"tensor's device is :{varTorch.device}")
    print(f"tensor's max  num is :{varTorch.max()}")
    print(f"tensor's mean num is :{varTorch.mean()}")
    print(f"tensor's min  num is :{varTorch.min()}")
    # plt.hist(arrayNp.ravel(), bins='auto')
    # plt.ylim(10000)
    # plt.hist(arrayNp.ravel(), bins=255)
    # plt.show()
    # plt.close()





# y add
def event_stream_to_voxel_frames(
        ev_npz_path_list: List[str],
        width: int, 
        height: int,
):
    # print_list(ev_npz_path_list)


    if ev_npz_path_list ==[]:
        events_frames = np.zeros((16, height, width))
    else:
        tmp = []
        for i in ev_npz_path_list:
            ev_n4_i = load_ev_npz_erf(i)
            tmp.append(ev_n4_i)
        
        ev_n4 = np.concatenate(tmp, axis=0).astype(np.float)  # n,4
        # print(f"==>> ev_n4.shape: {ev_n4.shape}")

        if ev_n4.shape[0] > 0:
            events_frames = events_to_voxel_grid(ev_n4,num_bins=16, width=width, height=height ) #  (16, 975, 1440)    # , device = torch.device('cuda')
        else:
            events_frames = np.zeros((16, height, width))


    # print(f"==>> np.unique(events_frames): {np.unique(events_frames)}")
    # ts.show(events_frames[:,None,:,:])

    return events_frames



def events_to_voxel_grid(events, num_bins, width, height):
    """
    adapted from rpg_e2vid/events_to_voxel_grid.py events_to_voxel_grid
    """
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()   # num_bins*height*width

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp


    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    t = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = t.astype(int)
    dts = t - tis
    vals_left = pols * (1.0 - dts)  # 距离加权
    vals_right = pols * dts         # 距离加权

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])
    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid  # mhw



def events_to_voxel_grid_pytorch_old240415(events, num_bins, width, height, device):
    """
    adapted from rpg_e2vid/events_to_voxel_grid.py events_to_voxel_grid_pytorch
    y comment: events_to_voxel_grid_pytorch 的耗时大约是 events_to_voxel_grid 的 1/3
    events: 输入前应该，将dtype 设置为 float, 如果是int,输出的voxel_grid： mhw, 的dtype也会是int，0,1,2之类，不对
    """
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(device)

        with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            t = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(t)
            tis_long = tis.long()
            dts = t - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices]
                                  * width + tis_long[valid_indices] * width * height,
                                  source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=xs[valid_indices] + ys[valid_indices] * width
                                  + (tis_long[valid_indices] + 1) * width * height,
                                  source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid  # mhw



def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    adapted from rpg_e2vid/events_to_voxel_grid.py events_to_voxel_grid_pytorch
    y comment: events_to_voxel_grid_pytorch 的耗时大约是 events_to_voxel_grid 的 1/3
    events: 输入前应该，将dtype 设置为 float, 如果是int,输出的voxel_grid： mhw, 的dtype也会是int，0,1,2之类，不对
    """
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    # DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events)
        # with DeviceTimer('Events -> Device (voxel grid)'):
        events_torch = events_torch.to(device)

        # with DeviceTimer('Voxel grid voting'):
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        t = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(t)
        tis_long = tis.long()
        dts = t - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices]
                                * width + tis_long[valid_indices] * width * height,
                                source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices] * width
                                + (tis_long[valid_indices] + 1) * width * height,
                                source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid  # mhw




#* ev downsampling
def evresize_np(evt, scale=1/8):   # ,h=624,w=968
    # events = events[:, :, ::2, ::2]
    assert (1/scale) % 1 == 0 , print('Error: scale is not  1/int ')
    sc = int(1/scale)
    hr_events = torch.from_numpy(evt)
    # print(f"==>> hr_events.shape: {hr_events.shape}")   # 624,968,2
    # low_resolution = (int(625 // scale), int(970 // scale))  # 78 121
    # hl,wl = low_resolution
    # lr_events = F.interpolate(
    #     hr_events,
    #     size=low_resolution,
    #     mode="bilinear",
    #     align_corners=False,
    # )   
    # todo:  用F.interpolate() 有bug
    lr_events = hr_events[::sc,::sc,:]
    # get_numpy_info(lr_events)
    lr_events = lr_events.numpy()
    # print(f"==>> lr_events.shape: {lr_events.shape}")  # 78,121,2
    return lr_events

def load_ev_npz_erf_old(path):
    '''
    
    '''
    with np.load(path) as data:
        t = data['t']
        x = np.round(data['x']/128)
        y = np.round(data['y']/128)
        p = data['p'] 
    # assume we have the raw event data in the arrays x, y, t, p
    w, h = 1440, 975

    # convert to signed integers
    x = x.astype(np.int16)
    y = y.astype(np.int16)
    # THE FOLLOWING 'MAGIC NUMBERS' ARE JUST GUESSES!
    # VISUALLY THE EVENTS SEEM TO BE ALIGNED WITH THE IMAGE
    # BUT THIS IS NOT CONFIRMED BY THE DATASET AUTHORS!
    # VALIDATE FOR YOURSELF BEFORE USING THIS!

    ev_n4 =  np.stack((t,x,y,p),axis=1)
    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:float 0~969  y:float 0~624    p:0,1   



def load_ev_npz_gopro(path,key_t = 't', key_x = 'x', key_y = 'y', key_p = 'p'):
    '''
    
    '''
    with np.load(path) as data:
        t = data[key_t]
        x = data[key_x]
        y = data[key_y]
        p = data[key_p] 

    # convert to signed integers
    x = x.astype(np.int16)
    y = y.astype(np.int16)

    ev_n4 =  np.stack((t,x,y,p),axis=1)
    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:float 0~1439  y:float 0~974    p:0,1   


def load_ev_npz_erf(path,key_t = 't', key_x = 'x', key_y = 'y', key_p = 'p'):
    '''
    
    '''
    with np.load(path) as data:
        t = data[key_t]
        x = np.round(data[key_x]/128)
        y = np.round(data[key_y]/128)
        p = data[key_p] 
    # assume we have the raw event data in the arrays x, y, t, p
    w, h = 1440, 975

    # convert to signed integers
    x = x.astype(np.int16)
    y = y.astype(np.int16)
    # THE FOLLOWING 'MAGIC NUMBERS' ARE JUST GUESSES!
    # VISUALLY THE EVENTS SEEM TO BE ALIGNED WITH THE IMAGE
    # BUT THIS IS NOT CONFIRMED BY THE DATASET AUTHORS!
    # VALIDATE FOR YOURSELF BEFORE USING THIS!

    ev_n4 =  np.stack((t,x,y,p),axis=1)
    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:float 0~1439  y:float 0~974    p:0,1   


def load_ev_npz_ced(path,key_t = 't', key_x = 'x', key_y = 'y', key_p = 'p'):
    '''
    
    '''
    with np.load(path) as data:
        t = data[key_t]
        x = data[key_x]
        y = data[key_y]
        p = data[key_p] 
    # assume we have the raw event data in the arrays x, y, t, p
    w, h = 1440, 975

    # convert to signed integers
    x = x.astype(np.int16)
    y = y.astype(np.int16)
    # THE FOLLOWING 'MAGIC NUMBERS' ARE JUST GUESSES!
    # VISUALLY THE EVENTS SEEM TO BE ALIGNED WITH THE IMAGE
    # BUT THIS IS NOT CONFIRMED BY THE DATASET AUTHORS!
    # VALIDATE FOR YOURSELF BEFORE USING THIS!

    ev_n4 =  np.stack((t,x,y,p),axis=1)
    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:int   y:int    p:0,1   


def load_ev_npz_bsergb_evreal(event_file_path):
    '''
    code from: https://github.com/ercanburak/EVREAL/tree/main
    '''
    def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
        data = data.astype(np.int32)
        overflow_indices = np.where(data > upper_limit*32)
        num_overflows = overflow_indices[0].shape[0]
        if fix_overflows and num_overflows > 0:
            data[overflow_indices] = data[overflow_indices] - 65536
        data = data / 32.0
        data = np.rint(data)
        data = data.astype(np.int16)
        data = np.clip(data, 0, upper_limit)
        return data

    FRAME_WIDTH = 970
    FRAME_HEIGHT = 625

    event_file_data = np.load(event_file_path)
    x = convert_and_fix_event_pixels(event_file_data['x'], FRAME_WIDTH - 1).astype(np.int16)   # int16 0-969
    y = convert_and_fix_event_pixels(event_file_data['y'], FRAME_HEIGHT - 1).astype(np.int16)  # int16 0-624
    t = event_file_data['timestamp'].astype(np.int64)    # float64: microsecond 
    p = event_file_data['polarity'].astype(np.int16)                        # uint8: 0,1

    # t = t / 1000000.0  # convert us to s
    # print(f"==>> t.dtype: {t.dtype}")
    # print(f"==>> x.dtype: {x.dtype}")
    # print(f"==>> y.dtype: {y.dtype}")
    # print(f"==>> p.dtype: {p.dtype}")
    ev_n4 =  np.stack((t,x,y,p),axis=1)  # n,4    t,x,y,p    t:microsecond    x:float 0~969  y:float 0~624    p:0,1   
    # print(f"==>> ev_n4.shape: {ev_n4.shape}")

    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:float 0~969  y:float 0~624    p:0,1   


def ev_n4_l_to_events_frames(ev_n4_l,num_bins,height,width):

    if ev_n4_l ==[]:
        events_frames = np.zeros((num_bins, height, width))
    else:
        ev_n4 = np.concatenate(ev_n4_l, axis=0).astype(float)
        
        # logger.debug(f"==>> ev_n4: {ev_n4}")
        if ev_n4.shape[0] > 0:
            load_ev_voxel_in_cuda=False
            if load_ev_voxel_in_cuda:
                events_frames_cuda = events_to_voxel_grid_pytorch(ev_n4,num_bins=num_bins, width=width, height=height, device = torch.device('cuda:0') )  # 'cuda:0'     # , device = torch.device('cuda')
                events_frames = events_frames_cuda.cpu().numpy()
                del events_frames_cuda
            else:
                events_frames = events_to_voxel_grid(ev_n4,num_bins=num_bins, width=width, height=height)  # 'cuda:0'     # , device = torch.device('cuda')

        else:
            events_frames = np.zeros((num_bins, height, width))
    # logger.debug("".center(50, "-"))
    # for i in range(16):

        # logger.debug(f"==>> events_frames[{i},:,:].max(),min(): {events_frames[i,:,:].max()}, {events_frames[i,:,:].min()}")
        # logger.debug(f"==>> len(np.unique(events_frames[i,:,:])): {len(np.unique(events_frames[i,:,:]))}")
    return events_frames # ([16, 975, 1440])



def ev_downsample(crop_events,lr_h ,lr_w ):
    '''
    adapted from /home/hjy/workspace/interp_photo/04_inr/41_INR-Event-VSR_egvsr/egvsr/datasets/alpx_vsr_dataset.py
    crop_events:
        ([36, 1, 1024, 1024])

    rtn:
        ([36, 1, 120, 180])
    '''
    import torch.nn.functional as F
    # print(f"==>> crop_events.shape: {crop_events.shape}")
    lr_events = F.interpolate(
        crop_events,              # ([36, 1, 1024, 1024])
        size=(lr_h, lr_w),               #  (128,128)
        mode="bilinear",
        align_corners=False,
        )
    # print(f"==>> lr_events.shape: {lr_events.shape}")
    # logger.debug(f"==>> crop_events.shape: {crop_events.shape}")
    # logger.debug(f"==>> lr_events.shape: {lr_events.shape}")

    return lr_events
    


def load_ev_npz_bsergb(path):
    '''
    code from: https://github.com/uzh-rpg/timelens-pp/issues/2
    '''
    with np.load(path) as data:
        t = data['timestamp']
        x = data['x']
        y = data['y']
        p = data['polarity'] 
    # assume we have the raw event data in the arrays x, y, t, p
    w, h = 970, 625

    # convert to signed integers
    x = x.astype(np.int16)
    y = y.astype(np.int16)
    # THE FOLLOWING 'MAGIC NUMBERS' ARE JUST GUESSES!
    # VISUALLY THE EVENTS SEEM TO BE ALIGNED WITH THE IMAGE
    # BUT THIS IS NOT CONFIRMED BY THE DATASET AUTHORS!
    # VALIDATE FOR YOURSELF BEFORE USING THIS!
    max_val_allowed_x = round((w / h) * 20000)
    max_val_allowed_y = 20000


    # filter out events that are outside the image
    mask = (x >= 0) & (x <= max_val_allowed_x) & (y >= 0) & (y <= max_val_allowed_y)

    x = x[mask]
    y = y[mask]
    t = t[mask]
    p = p[mask]

    # convert the events to pixel coordinates
    x = (x / max_val_allowed_x * (w-1)).astype(float)
    y = (y / max_val_allowed_y * (h-1)).astype(float)

    ev_n4 =  np.stack((t,x,y,p),axis=1)
    return ev_n4  # n,4    t,x,y,p    t:microsecond    x:float 0~969  y:float 0~624    p:0,1   





def load_ev_npz_bsergb_evreal_old(event_file_path):
    '''
    code from: https://github.com/ercanburak/EVREAL/tree/main
    '''
    def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
        data = data.astype(np.int32)
        overflow_indices = np.where(data > upper_limit*32)
        num_overflows = overflow_indices[0].shape[0]
        if fix_overflows and num_overflows > 0:
            data[overflow_indices] = data[overflow_indices] - 65536
        data = data / 32.0
        data = np.rint(data)
        data = data.astype(np.int16)
        data = np.clip(data, 0, upper_limit)
        return data

    FRAME_WIDTH = 970
    FRAME_HEIGHT = 625

    event_file_data = np.load(event_file_path)
    x_data = convert_and_fix_event_pixels(event_file_data['x'], FRAME_WIDTH - 1)   # int16 0-969
    y_data = convert_and_fix_event_pixels(event_file_data['y'], FRAME_HEIGHT - 1)  # int16 0-624
    t_data = event_file_data['timestamp'].astype(np.float64)    # float64: microsecond(us)
    p_data = event_file_data['polarity']                        # uint8: 0,1

    ev_n4 =  np.stack((t_data,x_data,y_data,p_data),axis=1)  # n,4    t,x,y,p    t:microsecond    x:float 0~969  y:float 0~624    p:0,1   

    return ev_n4  # n,4     t:microsecond    x:float 0~969  y:float 0~624    p:0,1   
