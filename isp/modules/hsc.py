import math

import torch

from isp.modules import ISPModule


class HSC(ISPModule):
    """ Hue & Saturation Correction
    - Input:
        YUV 8 Bit
    - Output:
        YUV 8 Bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        hue = exif['hsc']['hue']
        saturation = exif['hsc']['saturation']

        bit_in = 8
        bit_out = 8

        return {
            'hue': hue,
            'saturation': saturation,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        hue_sin = math.sin(param['hue'] * math.pi / 180.0)
        hue_cos = math.cos(param['hue'] * math.pi / 180.0)

        saturation = param['saturation']

        Y = signal_in[:, 0:1, :, :]
        U = signal_in[:, 1:2, :, :]
        V = signal_in[:, 2:3, :, :]

        U = (U - 128.0) * hue_cos + (V - 128.0) * hue_sin
        V = (V - 128.0) * hue_cos + (V - 128.0) * hue_sin

        U = saturation * U + 128
        V = saturation * V + 128

        signal_out = torch.cat([Y, U, V], 1)

        signal_out = torch.clamp(signal_out, 0, 255)
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
