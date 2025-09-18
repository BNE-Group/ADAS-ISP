import torch.nn as nn


class ISPModule(nn.Module):
    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        raise NotImplementedError

    def process(self, signal, param=None):
        raise NotImplementedError

    def forward(self, signal_in, exif=None, stat=None):
        param = self.parse_param(exif, stat)
        signal_out = self.process(signal_in, param)
        return signal_out, param


# RAW
from isp.modules.dpc import DPC
from isp.modules.blc import BLC
from isp.modules.lsc import LSC
from isp.modules.aaf import AAF
from isp.modules.awb import AWB
from isp.modules.dem import DEM

# RGB
from isp.modules.ccm import CCM
from isp.modules.gamma import Gamma

from isp.modules.yuv2rgb import YUV2RGB
from isp.modules.rgb2yuv import RGB2YUV

# YUV
from isp.modules.hsc import HSC
from isp.modules.bcc import BCC

__all__ = [
    'DPC', 'BLC', 'LSC', 'AAF', 'AWB', 'DEM',
    'CCM', 'Gamma', 'YUV2RGB', 'RGB2YUV',
    'HSC', 'BCC', 'ISPModule'
]
