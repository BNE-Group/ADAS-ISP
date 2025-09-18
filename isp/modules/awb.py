import torch

from isp.modules import ISPModule


class AWB(ISPModule):
    """ Auto White Balance
    - Input:
        Raw 10 bit
    - Output:
        Raw 10 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        r_gain = exif['awb']['r_gain']
        gr_gain = exif['awb']['gr_gain']
        gb_gain = exif['awb']['gb_gain']
        b_gain = exif['awb']['b_gain']

        bit_in = 10
        bit_out = 10

        return {
            'r_gain': r_gain,
            'b_gain': b_gain,
            'gr_gain': gr_gain,
            'gb_gain': gb_gain,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        signal_in = signal_in / (2 ** param['bit_in'])

        r_gain = param['r_gain']
        gr_gain = param['gr_gain']
        gb_gain = param['gb_gain']
        b_gain = param['b_gain']

        signal_out = torch.zeros_like(signal_in)

        # RGGB pattern
        R = signal_in[..., 0::2, 0::2]
        Gr = signal_in[..., 0::2, 1::2]
        Gb = signal_in[..., 1::2, 0::2]
        B = signal_in[..., 1::2, 1::2]

        signal_out[:, :, 0::2, 0::2] = R * r_gain
        signal_out[:, :, 0::2, 1::2] = Gr * gr_gain
        signal_out[:, :, 1::2, 0::2] = Gb * gb_gain
        signal_out[:, :, 1::2, 1::2] = B * b_gain

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
