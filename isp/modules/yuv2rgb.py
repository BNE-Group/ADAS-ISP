import torch

from isp.modules import ISPModule


class YUV2RGB(ISPModule):
    """ YUV to RGB
    - Input:
        YUV 8 Bit
    - Output:
        RGB 8 Bit
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
        Y = signal_in[:, 0:1, :, :]
        U = signal_in[:, 1:2, :, :]
        V = signal_in[:, 2:3, :, :]

        R = 1.164 * (Y - 16) + 1.596 * (V - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.392 * (U - 128)
        B = 1.164 * (Y - 16) + 2.017 * (U - 128)

        signal_out = torch.cat([R, G, B], dim=1)

        signal_out = torch.clamp(signal_out, 0, 255)

        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
