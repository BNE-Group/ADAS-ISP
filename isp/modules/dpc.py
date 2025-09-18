import torch
import torch.nn.functional as F

from isp.modules import ISPModule


class DPC(ISPModule):
    """ Dead Pixel Correction
    - Input:
        Raw 10 bit
    - Output:
        Raw 10 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        threshold = exif['dpc']['threshold']
        bit_in = 10
        bit_out = 10

        return {
            'threshold': threshold,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])  # 0~1023 -> 0~1

        # calculate neighbors
        signal_in_padding = F.pad(signal_in, (2, 2, 2, 2), mode='reflect')

        p00 = signal_in_padding[..., 0:H + 0, 0:W + 0]
        p01 = signal_in_padding[..., 0:H + 0, 2:W + 2]
        p02 = signal_in_padding[..., 0:H + 0, 4:W + 4]
        p10 = signal_in_padding[..., 2:H + 2, 0:W + 0]
        p12 = signal_in_padding[..., 2:H + 2, 4:W + 4]
        p20 = signal_in_padding[..., 4:H + 4, 0:W + 0]
        p21 = signal_in_padding[..., 4:H + 4, 2:W + 2]
        p22 = signal_in_padding[..., 4:H + 4, 4:W + 4]

        d00 = torch.abs(p00 - signal_in)
        d01 = torch.abs(p01 - signal_in)
        d02 = torch.abs(p02 - signal_in)
        d10 = torch.abs(p10 - signal_in)
        d12 = torch.abs(p12 - signal_in)
        d20 = torch.abs(p20 - signal_in)
        d21 = torch.abs(p21 - signal_in)
        d22 = torch.abs(p22 - signal_in)

        # calculate outliers
        signal_count = torch.zeros_like(signal_in)
        threshold = param['threshold'] / (2 ** param['bit_in'])  # 0~1023 -> 0~1

        signal_count = torch.where(d00 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d01 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d02 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d10 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d12 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d20 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d21 > threshold, signal_count, signal_count + 1)
        signal_count = torch.where(d22 > threshold, signal_count, signal_count + 1)

        # mean outliers
        signal_out = torch.where(signal_count == 8, (p01 + p10 + p12 + p21) / 4.0, signal_in)

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
