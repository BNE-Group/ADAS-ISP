import torch

from isp.modules import ISPModule


class Gamma(ISPModule):
    """
    Gamma Correction
    - Input:
        RGB 8 bit
    - Output:
        RGB 8 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        gamma = exif['gamma']['factor']
        bit_in = 8
        bit_out = 8

        return {
            'gamma': gamma,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])

        # RGB space
        signal_out = torch.pow(signal_in, param['gamma'])

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
