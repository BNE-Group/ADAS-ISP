import torch.nn as nn

from isp.exif import parse_exif_json
from isp.modules import *
from isp.raw import parse_raw_torch


class ISPPipeline(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

        self.raw = None
        self.w = None
        self.h = None
        self.exif = None
        self.rgb = None

        self.device = device

        self.pipeline = nn.ModuleList([
            DPC(), BLC(), LSC(), AAF(), AWB(), DEM(), CCM(), Gamma(), RGB2YUV(), HSC(), BCC(), YUV2RGB()
        ]).to(self.device)

    def load_data(self, raw_path, exif_path):
        self.exif = parse_exif_json(exif_path)
        self.w = self.exif['width']
        self.h = self.exif['height']
        self.raw = parse_raw_torch(raw_path, self.h, self.w).to(self.device)

    def process(self):
        if self.raw is None:
            print('Please load raw data first. Using load_data().')
            return None

        signal_in = self.raw
        signal_out = self.raw
        exif = self.exif

        for i in range(len(self.pipeline)):
            signal_out, _ = self.pipeline[i](signal_in, exif)
            signal_in = signal_out

        self.rgb = signal_out

    def get_rgb_numpy(self):
        if self.rgb is None:
            print('Please load raw data first. Using load_data().')
            return None

        return self.rgb.detach().to('cpu').numpy()[0].transpose(1, 2, 0)

    def get_rgb_torch(self):
        if self.rgb is None:
            print('Please load raw data first. Using load_data().')
            return None

        return self.rgb
