
#* y add
"""
adapted from: 
https://github.com/intelpro/CBMNet/blob/main/models/final_models/submodules.py
"""
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError
class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
               op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out

def conv3x3_leaky_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True), nn.LeakyReLU(0.1))


def conv_resblock_two(in_channels, out_channels, stride=1): 
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels))




def tconv4x4(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True)

def tconv_relu(in_channels, out_channels, stride=1): 
    return nn.Sequential(tconv4x4(in_channels, out_channels, stride), nn.ReLU())







class event_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(event_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two(nf, nf)
        self.conv2 = conv_resblock_two(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two(2*nf, 4*nf, stride=2)


        # 创建转置卷积层
        self.tconv3 = tconv_relu(
            in_channels=4*nf, out_channels=2*nf,  stride=2 
        )
        self.tconv2 = tconv_relu(
            in_channels=2*nf, out_channels=nf,  stride=2
        )
        self.tconv1 = tconv_relu(
            in_channels=nf, out_channels=nf,  stride=2
        )

        
    def forward(self, x):   # 1,16,32,32
        x_ =  self.conv0(x) # 1,16,32,32
        f1 = self.conv1(x_) # 1,16,32,32
        f2 = self.conv2(f1)     # 1,32,16,16    f2
        f3 = self.conv3(f2)         # 1,64,8,8  f3
 
        f4 = self.tconv3(f3)    # 1,32,16,16
        f5 = self.tconv2(f4) # 1,16,32,32
        f6 = self.tconv1(f5) # 1,16,32,32


        # return [f1, f2, f3, f4, f5]
        return f6      # 1,16,32,32