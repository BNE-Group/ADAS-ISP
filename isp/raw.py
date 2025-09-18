import numpy as np
import torch


def parse_raw_numpy(raw_file, h=None, w=None):
    raw_data = np.fromfile(raw_file, dtype='uint16', sep='')

    if h is not None and w is not None:
        raw_data = raw_data.reshape(h, w)

    return raw_data


def parse_raw_torch(raw_file, h, w):
    raw_data = np.fromfile(raw_file, dtype='uint16', sep='')
    raw_data = raw_data.reshape(h, w)
    raw_data = torch.FloatTensor(raw_data[np.newaxis, np.newaxis])

    return raw_data
