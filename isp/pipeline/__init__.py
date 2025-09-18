import torch.nn as nn


class ISPPipeline(nn.Module):
    def __init__(self):
        super().__init__()

    def process(self, signal, exif):
        raise NotImplementedError
