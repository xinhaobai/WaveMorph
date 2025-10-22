#
# reference: https://github.com/pfriedri/wdm-3d/blob/main/DWT_IDWT/DWT_IDWT_layer.py
# https://github.com/pfriedri/wdm-3d

"""
自定义 pytorch 层，实现一维、二维、三维张量的 DWT 和 IDWT，未考虑边界延拓
只有当图像行列数都是偶数，且重构滤波器组低频分量长度为 2 时，才能精确重构，否则在边界处有误差。
"""
import math

import numpy as np
import pywt
import torch
from torch.nn import Module

from models_WaveMorph.layer_DWT_IDWT_Functions import  DWTFunction_1D, IDWTFunction_1D, \
    DWTFunction_2D_tiny, DWTFunction_2D, IDWTFunction_2D, \
    DWTFunction_3D, IDWTFunction_3D


__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D',
           'IDWT_2D', 'DWT_3D', 'IDWT_3D', 'DWT_2D_tiny']


class DWT_1D(Module):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    """

    def __init__(self, wavename):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        用于序列分解的一维离散小波变换 DWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1):end]
        matrix_g = matrix_g[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, input):
        """
        input_low_frequency_component = \mathcal{L} * input
        input_high_frequency_component = \mathcal{H} * input
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """

    def __init__(self, wavename):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        用于序列重构的一维离散小波逆变换 IDWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        生成变换矩阵
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half - 1):end]
        matrix_g = matrix_g[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D_tiny(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              #hfc_lh: (N, C, H/2, W/2)
              #hfc_hl: (N, C, H/2, W/2)
              #hfc_hh: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
    the all four components could be get using DWT_2D, which is used in WaveUNet.
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        #input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        #input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        #input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """

    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        recontructing the original 2D data
        the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                             + \mathcal{H}^T * hfc_lh * \mathcal{L}
                             + \mathcal{L}^T * hfc_hl * \mathcal{H}
                             + \mathcal{H}^T * hfc_hh * \mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(
            HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_3D(Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_3D.apply(input,
                                    self.matrix_low_0.to(input.device),
                                    self.matrix_low_1.to(input.device),
                                    self.matrix_low_2.to(input.device),
                                    self.matrix_high_0.to(input.device),
                                    self.matrix_high_1.to(input.device),
                                    self.matrix_high_2.to(input.device))


class IDWT_3D(Module):
    """
    input:  lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    output: the original 3D data -- (N, C, D, H, W)
    """

    def __init__(self, wavename):
        """
        3D inverse DWT (IDWT) for 3D data reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        """
        :param LLL: the low-frequency component, lfc
        :param LLH: the high-frequency componetn, hfc_llh
        :param LHL: the high-frequency componetn, hfc_lhl
        :param LHH: the high-frequency componetn, hfc_lhh
        :param HLL: the high-frequency componetn, hfc_hll
        :param HLH: the high-frequency componetn, hfc_hlh
        :param HHL: the high-frequency componetn, hfc_hhl
        :param HHH: the high-frequency componetn, hfc_hhh
        :return: the original 3D input data
        """
        assert len(LLL.size()) == len(LLH.size()) == len(
            LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(
            HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width = LLL.size()[-1] + HHH.size()[-1]
        self.get_matrix()
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0.to(LLL.device),
                                     self.matrix_low_1.to(LLL.device),
                                     self.matrix_low_2.to(LLL.device),
                                     self.matrix_high_0.to(LLL.device),
                                     self.matrix_high_1.to(LLL.device),
                                     self.matrix_high_2.to(LLL.device))


# =============================================================
# TODO: XH
# =============================================================
# Referenced from: https://github.com/YehLi/ImageNetModel/blob/main/classification/wavevit.py
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models_WaveMorph.layer_nn_forDWT import checkpoint, conv_nd, linear,\
    avg_pool_nd, zero_module, normalization, timestep_embedding



class Upsample(nn.Module):
    """
    A wavelet upsampling layer with an optional convolution on the skip connections used to perform upsampling.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d

        self.use_freq = use_freq
        self.idwt = IDWT_3D("haar")

        # Grouped convolution on 7 high frequency subbands (skip connections)
        if use_conv:
            self.conv = conv_nd(dims, self.channels * 7, self.out_channels * 7, 3, padding=1, groups=7)

    def forward(self, x):
        if isinstance(x, tuple):
            skip = x[1]
            x = x[0]
        assert x.shape[1] == self.channels

        if self.use_conv:
            skip = self.conv(torch.cat(skip, dim=1) / 3.) * 3.
            skip = tuple(torch.chunk(skip, 7, dim=1))

        if self.use_freq:
            x = self.idwt(3. * x, skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6])
        else:
            if self.dims == 3 and self.resample_2d:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")

        return x, None


class Downsample(nn.Module):
    """
    A wavelet downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        self.use_freq = use_freq
        self.dwt = DWT_3D("haar")

        stride = (1, 2, 2) if dims == 3 and resample_2d else 2

        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        elif self.use_freq:
            self.op = self.dwt
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.use_freq:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.op(x)
            x = (LLL / 3., (LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        else:
            x = self.op(x)
        return x


class WaveletDownsample(nn.Module):
    """
    Implements the wavelet downsampling blocks used to generate the input residuals.

    :param in_ch: number of input channels.
    :param out_ch: number of output channels (should match the feature size of the corresponding U-Net level)
    """
    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = conv_nd(3, self.in_ch * 8, self.out_ch, 3, stride=1, padding=1)
        self.dwt = DWT_3D('haar')

    def forward(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        x = torch.cat((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), dim=1) / 3.
        return self.conv(x)



class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D("haar")
        # self.reduce = nn.Sequential(
        #     nn.Conv3d(dim, dim//4, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm3d(dim//4),
        #     nn.ReLU(inplace=True),
        # )
        # self.reduce = WaveletDownsample(in_ch=dim//8, out_ch=dim//8) #
        self.filter = nn.Sequential(
            nn.Conv3d(dim//8, dim//8, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm3d(dim//8),
            nn.ReLU(inplace=True),
        )
        # self.kv_embed = nn.Conv3d(dim//8, dim//8, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        # self.q = nn.Linear(dim, dim)
        self.qkv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3)
        )
        self.proj = nn.Linear(dim+dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W, T):
        B, N, C = x.shape
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # x = x.view(B, H, W, T, C).permute(0, 4, 1, 2, 3)
        x = x.view(B, H * 2, W * 2, T * 2, -1).permute(0, 4, 1, 2, 3)
        # x = self.reduce(x) # x.reduce.shape:  torch.Size([512, 24, 5, 6, 7])
        # print("x.reduce.shape: ", x.shape)
        x_dwt_out = self.dwt(x)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH \
                    = x_dwt_out[0], x_dwt_out[1],x_dwt_out[2],\
                      x_dwt_out[3],x_dwt_out[4],x_dwt_out[5],\
                      x_dwt_out[6],x_dwt_out[7]

        # print("x.dwt.shape: ", x_dwt.shape)
        LLL = self.filter(LLL) # torch.Size([512, 12, 5, 6, 7])
        LLH = self.filter(LLH)
        LHL = self.filter(LHL)
        LHH = self.filter(LHH)
        HLL = self.filter(HLL)
        HLH = self.filter(HLH)
        HHL = self.filter(HHL)
        HHH = self.filter(HHH)
        # print("x.filter.shape: ", x_dwt.shape)

        # x_dwt = torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)  #
        x_idwt = self.idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)

        x_idwt = x_idwt.view(B, -1, C)#.transpose(1, 2)
        # print("x_idwt.transpose: ", x_idwt.shape)
        # qkv = self.qkv_embed(x_dwt).reshape(B, C//8, -1).permute(0, 2, 1)
        qkv = self.qkv(x_idwt).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x.q*k*v.shape: ", x.shape)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x



# 2024/10/29 using
class Wave_WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim+dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # wave
        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D("haar")
        # self.reduce = WaveletDownsample(in_ch=dim//8, out_ch=dim//8) #
        self.filter = nn.Sequential(
            nn.Conv3d(dim//8, dim//8, kernel_size=3, padding=1, stride=1, groups=1),
            nn.InstanceNorm3d(dim//8),
            # nn.BatchNorm3d(dim//8),#org
            nn.ReLU(inplace=True),
        )

    def forward(self, x, H,W,T, mask=None):

        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = x.view(B_, H * 2, W * 2, T * 2, -1).permute(0, 4, 1, 2, 3)
        # x = self.reduce(x) # x.reduce.shape:  torch.Size([512, 24, 5, 6, 7])
        x_dwt_out = self.dwt(x)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH \
            = x_dwt_out[0], x_dwt_out[1], x_dwt_out[2], \
              x_dwt_out[3], x_dwt_out[4], x_dwt_out[5], \
              x_dwt_out[6], x_dwt_out[7]

        LLL = self.filter(LLL)  # torch.Size([512, 12, 5, 6, 7])
        LLH = self.filter(LLH)
        LHL = self.filter(LHL)
        LHH = self.filter(LHH)
        HLL = self.filter(HLL)
        HLH = self.filter(HLH)
        HHL = self.filter(HHL)
        HHH = self.filter(HHH)

        x_idwt = self.idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)#-x
        x_idwt = x_idwt.view(B_, -1, C)  # .transpose(1, 2)

        x_dwt_all = torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        x_dwt_all = x_dwt_all.view(B_, -1, C)
        # print("x_dwt_all: ", x_dwt_all.shape)
        # print("LLL: ", LLL.shape)

        kv = self.kv(x_dwt_all).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q, kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = self.proj(torch.cat([x, x_idwt], dim=-1) )#org
        # x = self.proj( x + x_idwt )
        x = self.proj( x )
        x = self.proj_drop(x)
        return x





# todo: wave window attention with Group (not work)
class Wave_Group_WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # wave
        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D("haar")

        # self.reduce = WaveletDownsample(in_ch=dim, out_ch=dim//8) #
        self.filter = nn.Sequential(
            nn.Conv3d(dim//8//self.num_heads, dim//8//self.num_heads, kernel_size=3, padding=1, stride=1, groups=1),
            # nn.MaxPool3d(3, stride=1, padding=1), # todo: nn.MaxPool3d(3, stride=2, padding=1) xh
            nn.BatchNorm3d(dim//8//self.num_heads),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, H,W,T, mask=None):
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        # print("x.org.shape :{}".format(x.shape))
        # for group
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i in range(self.num_heads):
            if i>0:
                # print("feat :{}, featin_{}:{}".format(feat.shape, i, feats_in[i].shape))
                feat = feat + feats_in[i]
            q = self.q(feat).reshape(B_, N, 1, self.num_heads, C//self.num_heads//self.num_heads).permute(2, 0, 3, 1, 4)
            feat = feat.view(B_, H * 2, W * 2, T * 2, -1).permute(0, 4, 1, 2, 3)
            # x = self.reduce(x) # x.reduce.shape:  torch.Size([512, 24, 5, 6, 7])
            # print("x.shape:{}".format(x.shape) )

            x_dwt_out = self.dwt(feat)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH \
                = x_dwt_out[0], x_dwt_out[1], x_dwt_out[2], \
                  x_dwt_out[3], x_dwt_out[4], x_dwt_out[5], \
                  x_dwt_out[6], x_dwt_out[7]

            LLL = self.filter(LLL)  # torch.Size([512, 12, 5, 6, 7])
            LLH = self.filter(LLH)
            LHL = self.filter(LHL)
            LHH = self.filter(LHH)
            HLL = self.filter(HLL)
            HLH = self.filter(HLH)
            HHL = self.filter(HHL)
            HHH = self.filter(HHH)

            x_idwt = self.idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)#-feat
            x_idwt = x_idwt.view(B_, -1, C)  # .transpose(1, 2)

            x_dwt_all = torch.cat([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
            x_dwt_all = x_dwt_all.view(B_, -1, C)
            # print("x_dwt_all: ", x_dwt_all.shape)
            # print("LLL: ", LLL.shape)

            kv = self.kv(x_dwt_all).reshape(B_, N, 2, self.num_heads, C//self.num_heads//self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = q, kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            if self.rpe:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] * self.window_size[2],
                    self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
                attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            feat = (attn @ v).transpose(1, 2).reshape(B_, -1, C)
            # print("\nfeat_out.shape: ", feat.shape)
            # print("attn.shape: ", attn.shape)
            # print("v.shape: ", v.shape)

            # feat = torch.cat([feat, x_idwt], dim=-1)
            feat = torch.add( feat, x_idwt)

            # print("feat_out.shape: ", feat.shape)

            feats_out.append(feat)
        # print("torch.cat(feats_out, 1): ", torch.cat(feats_out, 1).shape)
        x = self.proj(torch.cat(feats_out, 1))
        x = self.proj_drop(x)


        return x



        # # todo; 将当前头的输出传递给下一个头
        # x_interaction = x.clone()
        # for i in range(1, self.num_heads):
        #     x_interaction[:, :, i * (C // self.num_heads):(i + 1) * (C // self.num_heads)] += x[:, :, (i - 1) * (
        #                 C // self.num_heads):i * (C // self.num_heads)]
        #
        # return x_interaction



if __name__ == '__main__':
    dwt = DWT_2D("haar")
    iwt = IDWT_2D("haar")
    x = torch.randn(3, 3, 24, 24).cuda()
    xll = x
    wavelet_list = []
    for i in range(3):
        xll, xlh, xhl, xhh = dwt(xll)
        wavelet_list.append([xll, xlh, xhl, xhh])

    # xll = wavelet_list[-1] * torch.randn(xll.shape)
    for i in range(2)[::-1]:
        xll, xlh, xhl, xhh = wavelet_list[i]
        xll = iwt(xll, xlh, xhl, xhh)
        print(xll.shape)

    print(torch.sum(x - xll))
    print(torch.sum(x - iwt(*wavelet_list[0])))