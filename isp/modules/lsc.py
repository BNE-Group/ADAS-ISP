import torch

from isp.modules import ISPModule


class LSC(ISPModule):
    """ Lens Shading Correction
    - Input:
        Raw 10 bit
    - Output:
        Raw 10 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        intensity = exif['lsc']['intensity']
        bit_in = 10
        bit_out = 10

        return {
            'intensity': intensity,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])

        intensity = param['intensity']

        tW = torch.linspace(0.0, W - 1, W).view(1, 1, 1, W).expand(N, -1, H, -1).to(signal_in.device)
        tH = torch.linspace(0.0, H - 1, H).view(1, 1, H, 1).expand(N, -1, -1, W).to(signal_in.device)
        R = torch.sqrt((tW - W // 2) ** 2 + (tH - H // 2) ** 2)
        R = (R - R.min()) / (R.max() - R.min())
        signal_out = signal_in * (1 + intensity * (R + 0.5))

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
