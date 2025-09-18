import torch
import torch.nn.functional as F

from isp.modules import ISPModule


class DEM(ISPModule):
    """ Demosaic
    - Input:
        Raw 10 bit
    - Output:
        RGB 8 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        bit_in = 10
        bit_out = 8

        return {
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])

        signal_out = torch.zeros((N, 3, H, W)).to(signal_in.device)

        # Raw to RGB
        signal_in_pad = F.pad(signal_in, (1, 1, 1, 1), mode='reflect')

        # R-pos
        # R
        signal_out[:, 0:1, 0::2, 0::2] = signal_in[:, :, 0::2, 0::2]
        # G
        G1 = signal_in_pad[:, :, 0:H + 0, 0:W + 0][:, :, 0::2, 1::2]
        G2 = signal_in_pad[:, :, 0:H + 0, 0:W + 0][:, :, 1::2, 0::2]
        G3 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 0::2, 1::2]
        G4 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 1::2, 0::2]
        signal_out[:, 1:2, 0::2, 0::2] = (G1 + G2 + G3 + G4) / 4
        # B
        B1 = signal_in_pad[:, :, 0:H + 0, 0:W + 0][:, :, 0::2, 0::2]
        B2 = signal_in_pad[:, :, 0:H + 0, 2:W + 2][:, :, 0::2, 0::2]
        B3 = signal_in_pad[:, :, 2:H + 2, 0:W + 0][:, :, 0::2, 0::2]
        B4 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 0::2, 0::2]
        signal_out[:, 2:3, 0::2, 0::2] = (B1 + B2 + B3 + B4) / 4

        # Gr-pos
        # R
        R1 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 0::2, 0::2]
        R2 = signal_in_pad[:, :, 1:H + 1, 2:W + 2][:, :, 0::2, 1::2]
        signal_out[:, 0:1, 0::2, 1::2] = (R1 + R2) / 2
        # G
        signal_out[:, 1:2, 0::2, 1::2] = signal_in[:, :, 0::2, 1::2]
        # B
        B1 = signal_in_pad[:, :, 0:H + 0, 2:W + 2][:, :, 0::2, 0::2]
        B2 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 0::2, 0::2]
        signal_out[:, 2:3, 0::2, 1::2] = (B1 + B2) / 2

        # Gb-pos
        # R
        R1 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 0::2, 0::2]
        R2 = signal_in_pad[:, :, 2:H + 2, 1:W + 1][:, :, 1::2, 0::2]
        signal_out[:, 0:1, 1::2, 0::2] = (R1 + R2) / 2
        # G
        signal_out[:, 1:2, 1::2, 0::2] = signal_in[:, :, 1::2, 0::2]
        # B
        B1 = signal_in_pad[:, :, 2:H + 2, 0:W + 0][:, :, 0::2, 0::2]
        B2 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 0::2, 0::2]
        signal_out[:, 2:3, 1::2, 0::2] = (B1 + B2) / 2

        # B-pos
        # R
        R1 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 0::2, 0::2]
        R2 = signal_in_pad[:, :, 1:H + 1, 2:W + 2][:, :, 0::2, 1::2]
        R3 = signal_in_pad[:, :, 2:H + 2, 1:W + 1][:, :, 1::2, 0::2]
        R4 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 1::2, 1::2]
        signal_out[:, 0:1, 1::2, 1::2] = (R1 + R2 + R3 + R4) / 4
        # G
        G1 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 0::2, 1::2]
        G2 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 0::2, 1::2]
        G3 = signal_in_pad[:, :, 1:H + 1, 1:W + 1][:, :, 1::2, 0::2]
        G4 = signal_in_pad[:, :, 2:H + 2, 2:W + 2][:, :, 1::2, 0::2]
        signal_out[:, 1:2, 1::2, 1::2] = (G1 + G2 + G3 + G4) / 4
        # B
        signal_out[:, 2:3, 1::2, 1::2] = signal_in[:, :, 1::2, 1::2]

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
