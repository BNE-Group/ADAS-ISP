import torch

from isp.modules import ISPModule


class CCM(ISPModule):
    """ Color Correction Matrix
    - Input:
        RGB 8 bit
    - Output:
        RGB 8 bit
    """

    def __init__(self):
        super().__init__()

    def parse_param(self, exif=None, stat=None):
        matrix = exif['ccm']['matrix']
        bit_in = 8
        bit_out = 8

        return {
            'matrix': matrix,
            'bit_in': bit_in,
            'bit_out': bit_out,
        }

    def process(self, signal_in, param=None):
        N, C, H, W = signal_in.shape
        signal_in = signal_in / (2 ** param['bit_in'])

        signal_out = torch.zeros_like(signal_in)

        # RGB space
        signal_out[:, 0:1, :, :] = signal_in[:, 0:1, :, :] * param['matrix'][0][0] + \
                                   signal_in[:, 1:2, :, :] * param['matrix'][1][0] + \
                                   signal_in[:, 2:3, :, :] * param['matrix'][2][0]

        signal_out[:, 1:2, :, :] = signal_in[:, 0:1, :, :] * param['matrix'][0][1] + \
                                   signal_in[:, 1:2, :, :] * param['matrix'][1][1] + \
                                   signal_in[:, 2:3, :, :] * param['matrix'][2][1]

        signal_out[:, 2:3, :, :] = signal_in[:, 0:1, :, :] * param['matrix'][0][2] + \
                                   signal_in[:, 1:2, :, :] * param['matrix'][1][2] + \
                                   signal_in[:, 2:3, :, :] * param['matrix'][2][2]

        signal_out = torch.clamp(signal_out, 0, 1)
        signal_out = signal_out * (2 ** param['bit_out'])
        # def ste_round(x):
        #     return torch.round(x) - x.detach() + x
        # signal_out = ste_round(signal_out)
        signal_out = torch.round(signal_out)

        return signal_out
