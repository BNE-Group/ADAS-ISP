import json


def parse_exif_json(json_file):
    exif = {}
    metadata = json.load(open(json_file, 'r'))

    print(metadata)

    # convert metadata to exif info
    exif['width'] = metadata['width']
    exif['height'] = metadata['height']

    # dead pixel correction
    exif['dpc'] = {}
    exif['dpc']['threshold'] = metadata['dpc']['threshold']  # 10 bit integer

    # black level correction
    exif['blc'] = {}
    exif['blc']['blc_r'] = metadata['blc']['blc_r']  # 10 bit integer
    exif['blc']['blc_gr'] = metadata['blc']['blc_gr']  # 10 bit integer
    exif['blc']['blc_gb'] = metadata['blc']['blc_gb']  # 10 bit integer
    exif['blc']['blc_b'] = metadata['blc']['blc_b']  # 10 bit integer
    exif['blc']['r_gain'] = metadata['blc']['r_gain']  # float
    exif['blc']['b_gain'] = metadata['blc']['b_gain']  # float

    # lens shading correction
    exif['lsc'] = {}
    exif['lsc']['intensity'] = metadata['lsc']['intensity']  # float

    # anti-aliasing filter
    exif['aaf'] = {}

    # auto white balance
    exif['awb'] = {}
    exif['awb']['r_gain'] = metadata['awb']['r_gain']  # float
    exif['awb']['b_gain'] = metadata['awb']['b_gain']  # float
    exif['awb']['gr_gain'] = metadata['awb']['gr_gain']  # float
    exif['awb']['gb_gain'] = metadata['awb']['gb_gain']  # float

    # demosaic
    exif['demosaic'] = {}

    # color correction matrix
    exif['ccm'] = {}
    exif['ccm']['matrix'] = metadata['ccm']['matrix']  # 3x3 float

    # gamma correction
    exif['gamma'] = {}
    exif['gamma']['factor'] = metadata['gamma']['factor']  # float

    # rgb2yuv
    exif['rgb2yuv'] = {}

    # hue & saturation correction
    exif['hsc'] = {}
    exif['hsc']['hue'] = metadata['hsc']['hue']  # float
    exif['hsc']['saturation'] = metadata['hsc']['saturation']  # float

    # brightness & contrast correction
    exif['bcc'] = {}
    exif['bcc']['brightness'] = metadata['bcc']['brightness']  # float
    exif['bcc']['contrast'] = metadata['bcc']['contrast']  # float

    return exif
