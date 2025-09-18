import torch
import torch.nn.functional as F

from isp.modules import ISPModule


class AAF(ISPModule):
    """ Anti-Aliasing Filter
    - Input:
        Raw 10 bit
    - Output:
        Raw 10 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        bit_in = 10
        bit_out = 10

        return {
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])

        af_kernel = torch.tensor([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 8, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]
        ]).to(signal_in.device).view(1, 1, 5, 5) / 16.0
        singal_in_pad = F.pad(signal_in, (2, 2, 2, 2), mode='reflect')
        signal_out = F.conv2d(singal_in_pad, af_kernel, bias=None, stride=1, padding=0)

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
