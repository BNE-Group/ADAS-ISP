import cv2
from ultralytics import YOLO

from isp.pipeline.simple import ISPPipeline

if __name__ == '__main__':
    pipeline = ISPPipeline()
    pipeline.load_data('example/input.raw', 'example/exif.json')
    pipeline.process()
    rgb_torch = pipeline.get_rgb_torch()
    rgb_numpy = pipeline.get_rgb_numpy()

    rgb_numpy = cv2.cvtColor(rgb_numpy, cv2.COLOR_BGR2RGB)
    cv2.imshow('rgb', rgb_numpy / 255.0)
    cv2.waitKey()

    model = YOLO('yolo11n-seg.pt')
    results = model(rgb_torch)[0].show()
