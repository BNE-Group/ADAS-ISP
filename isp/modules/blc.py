import torch

from isp.modules import ISPModule


class BLC(ISPModule):
    """ Black Level Compensation
    - Input:
        Raw 10 bit
    - Output:
        Raw 10 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        blc_r = exif['blc']['blc_r']
        blc_gr = exif['blc']['blc_gr']
        blc_gb = exif['blc']['blc_gb']
        blc_b = exif['blc']['blc_b']

        r_gain = exif['blc']['r_gain']
        b_gain = exif['blc']['b_gain']

        bit_in = 10
        bit_out = 10

        return {
            'blc_r': blc_r,
            'blc_gr': blc_gr,
            'blc_gb': blc_gb,
            'blc_b': blc_b,
            'r_gain': r_gain,
            'b_gain': b_gain,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        signal_in = signal_in / (2 ** param['bit_in'])

        blc_r = param['blc_r'] / (2 ** param['bit_in'])
        blc_b = param['blc_b'] / (2 ** param['bit_in'])
        blc_gr = param['blc_gr'] / (2 ** param['bit_in'])
        blc_gb = param['blc_gb'] / (2 ** param['bit_in'])
        r_gain = param['r_gain']
        b_gain = param['b_gain']

        signal_out = torch.zeros_like(signal_in)

        # RGGB pattern
        R = signal_in[..., 0::2, 0::2]
        Gr = signal_in[..., 0::2, 1::2]
        Gb = signal_in[..., 1::2, 0::2]
        B = signal_in[..., 1::2, 1::2]

        R_out = R + blc_r
        B_out = B + blc_b
        Gr_out = Gr + blc_gr + R_out * r_gain
        Gb_out = Gb + blc_gb + B_out * b_gain

        signal_out[..., 0::2, 0::2] += R_out
        signal_out[..., 0::2, 1::2] += Gr_out
        signal_out[..., 1::2, 0::2] += Gb_out
        signal_out[..., 1::2, 1::2] += B_out

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
