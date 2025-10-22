import torch 
import torch.nn as nn
import torch.nn.functional as F
from models_WaveMorph.WiNet.nn_util import *
from models_WaveMorph.WiNet.utils import *
from models_WaveMorph.WiNet.DWT_IDWT.DWT_IDWT_Functions import *
# from monai.networks.layers.utils import get_act_layer
import torch.nn.functional as nnf

# referenced TransMorph xh

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)




# wavelet='haar'
# wavelet='db4'
class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, wavelet='haar', dwt_fn=1, up_train =True):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ch = start_channel
        act = "leakyrelu"
        # act = "prelu"
        # act =  ("leakyrelu", {"negative_slope": 0.2})  
     
        norm_enc  = None # batch | instance
        norm_dec  = None # batch | instance
        
        if dwt_fn == 1: 
            from models_WaveMorph.WiNet.DWT_IDWT.DWT_IDWT_layer  import DWT_3D,IDWT_3D
        else:
            from models_WaveMorph.WiNet.DWT_IDWT.DWT_IDWT_layer2 import DWT_3D,IDWT_3D

        super(UNet, self).__init__()

        self.DWT  =  DWT_3D(wavename=wavelet)
        self.iDWT = IDWT_3D(wavename=wavelet)
        
        self.layer1 =  conv_twice(8*self.in_channel, self.ch*4,  s=1,  act=act, norm='INSTANCE')
        
        self.layer2 =  conv_twice(self.ch*4, self.ch*4,  s=2,  act=act,norm=norm_enc) 
        self.layer3 =  conv_twice(self.ch*4, self.ch*8,  s=2,  act=act,norm=norm_enc)  
        self.layer4 =  conv_twice(self.ch*8, self.ch*8,  s=2,  act=act,norm=norm_enc)  
        
        self.up_conv1 = Up_conv(self.ch*8, self.ch*8, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*8)
        self.f_1_8 = self.to_out(self.ch*8, 8*3)
        
        self.up_conv2 = Up_conv(self.ch*8, self.ch*8, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*4)
        self.f_1_4 = self.to_out(self.ch*8, 7*3)
        
        self.up_conv3 = Up_conv(self.ch*8, self.ch*4, train=up_train, act=act, norm=norm_dec, skip_c=self.ch*4)
        self.f_1_2 = self.to_out(self.ch*4, 7*3)

        self.group_conv1 = nn.Sequential(nn.Conv3d(21*2,21*2,3,1,1, groups=7),
                                         get_act_layer(act),
                                         nn.Conv3d(21*2,21, 3,1,1,  groups=7),
                                         )
        self.group_conv2 = nn.Sequential(nn.Conv3d(21*2,21*2,3,1,1, groups=7),
                                         get_act_layer(act),
                                         nn.Conv3d(21*2,21, 3,1,1,  groups=7)
                                         )
        
        self.conv1 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
        self.conv2 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
        self.conv3 = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)

        # xh
        img_size = [128,128,128]
        # img_size = [96,96,96]
        # self.reg_head = RegistrationHead(in_channels=3, out_channels=3, kernel_size=3, )
        self.spatial_trans = SpatialTransformer(img_size)
       
    def to_out(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                )
        return layer

    def cat_h(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH,  LLH_,LHL_,LHH_,HLL_,HLH_,HHL_,HHH_, scale=2):
        LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.split_ch(self.up(torch.cat([LLH, LHL, LHH, HLL, HLH, HHL, HHH],dim=1), scale), 7)
        return torch.cat([LLH, LLH_, 
                          LHL, LHL_, 
                          LHH, LHH_,
                          HLL, HLL_, 
                          HLH, HLH_, 
                          HHL, HHL_, 
                          HHH, HHH_], dim=1)
        
    # def cat_h(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH,  
    #                 LLH_,LHL_,LHH_,HLL_,HLH_,HHL_,HHH_, scale=2):
    #     return torch.cat([self.up(LLH,scale), LLH_, 
    #                       self.up(LHL,scale), LHL_, 
    #                       self.up(LHH,scale), LHH_,
    #                       self.up(HLL,scale), HLL_, 
    #                       self.up(HLH,scale), HLH_, 
    #                       self.up(HHL,scale), HHL_, 
    #                       self.up(HHH,scale), HHH_], dim=1)
        
    
    def up(self, x, scale=2):
        return F.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=True) 
    
    def split_ch(self, a, n):
        return  [a[:,3*i:(3*i+3),...] for i in range(n)]



    # def forward(self, x, y, multi=False):
    def forward(self, x, multi=False):
        # e0 = torch.cat((*self.DWT(x), *self.DWT(y)), 1)
        source = x[:, 0:1, :, :] # xh
        e0 = self.DWT(x)
        E_ = torch.cat((e0[0],e0[1],e0[2],e0[3],
                        e0[4],e0[5],e0[6],e0[7]), 1)# xh#torch.Size([1, 8, 64, 64, 64])
        # print("Model.py: ", E_.shape)
        e1 = self.layer1(E_) # 64 64 48   | [80, 96, 112]
        e2 = self.layer2(e1) # 32 32 24   | [40, 48, 56]
        e3 = self.layer3(e2) # 16 16 12   | [20, 24, 28]
        e4 = self.layer4(e3) # 8 8 6      | [10, 12, 14]
                             # 4 4 3      | [5, 6, 7]
       
        d0 = self.up_conv1(e4, e3)
        idwt_1_8 = self.f_1_8(d0)
        
        d1 = self.up_conv2(d0, e2)
        idwt_1_4 = self.f_1_4(d1)
       
        d2 = self.up_conv3(d1, e1) 
        idwt_1_2 = self.f_1_2(d2)
       
        LLL_8, LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8 = self.split_ch(idwt_1_8, 8)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4        = self.split_ch(idwt_1_4, 7)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2        = self.split_ch(idwt_1_2, 7)
        
        xy_LLL_1_4 = self.conv1(self.iDWT(LLL_8, LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8))
     
        cat_value = self.cat_h(LLH_8,LHL_8,LHH_8,HLL_8,HLH_8,HHL_8,HHH_8,
                               LLH_4,LHL_4,LHH_4,HLL_4,HLH_4,HHL_4,HHH_4, scale=2)
    
        group_cov = self.group_conv1(cat_value)   
        LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_ = self.split_ch(group_cov, 7)
    
        xy_LLL_1_2 = self.conv2(self.iDWT(xy_LLL_1_4, LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_))
  
        cat_value2 = self.cat_h(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4, 
                                LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2, scale=2)
        
        group_cov2 = self.group_conv2(cat_value2)
        # print("group_cov2.shape: {}".format(group_cov2.shape))
        LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ = self.split_ch(group_cov2, 7)
        # print("self.iDWT(xy_LLL_1_2, LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ ): ",
        #       self.iDWT(xy_LLL_1_2, LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ ).shape)
        # ([1, 3, 128, 128, 128])

        xy_LLL_1_1 = self.conv3(self.iDWT(xy_LLL_1_2, LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_ ))
        f_xy = xy_LLL_1_1 # DVF
        # print(f'f_xy: {f_xy.shape}')  #torch.Size([1, 3, 128, 160, 128])
        
        # if multi:
        #     return  xy_LLL_1_1, xy_LLL_1_2, xy_LLL_1_4, [LLH_2_, LHL_2_, LHH_2_, HLL_2_, HLH_2_, HHL_2_, HHH_2_], [LLH_4_, LHL_4_, LHH_4_, HLL_4_, HLH_4_, HHL_4_, HHH_4_],[LLH_8, LHL_8, LHH_8, HLL_8, HLH_8, HHL_8, HHH_8]
        # print("f_xy.shape: {}".format(f_xy.shape))# torch.Size([1, 3, 128, 128, 128])

        out = self.spatial_trans(source, f_xy)

        return out, f_xy

    
