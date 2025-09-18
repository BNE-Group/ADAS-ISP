import torch

from isp.modules import ISPModule


class RGB2YUV(ISPModule):
    """ RGB to YUV
    - Input:
        RGB 8 Bit
    - Output:
        YUV 8 Bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        bit_in = 8
        bit_out = 8

        return {
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        R = signal_in[:, 0:1, :, :]
        G = signal_in[:, 1:2, :, :]
        B = signal_in[:, 2:3, :, :]

        Y = (0.257 * R + 0.504 * G + 0.098 * B) + 16
        U = (-0.148 * R - 0.291 * G + 0.439 * B) + 128
        V = (0.439 * R - 0.368 * G - 0.071 * B) + 128

        signal_out = torch.cat([Y, U, V], dim=1)

        signal_out = torch.clamp(signal_out, 0, 255)

        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
