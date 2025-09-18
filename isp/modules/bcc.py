import torch

from isp.modules import ISPModule


class BCC(ISPModule):
    """ Brightness & Contrast Control
    - Input:
        YUV 8 Bit
    - Output:
        YUV 8 Bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        brightness = exif['bcc']['brightness']
        contrast = exif['bcc']['contrast']

        bit_in = 8
        bit_out = 8

        return {
            'contrast': contrast,
            'brightness': brightness,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        contrast = param['contrast']
        brightness = param['brightness']

        Y = signal_in[:, 0:1, :, :]
        U = signal_in[:, 1:2, :, :]
        V = signal_in[:, 2:3, :, :]

        Y = Y + brightness
        Y = Y + (Y - 127.5) * contrast

        signal_out = torch.cat([Y, U, V], 1)

        signal_out = torch.clamp(signal_out, 0, 255)
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
