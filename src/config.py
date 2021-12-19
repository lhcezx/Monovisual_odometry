import os, sys
import argparse
from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("Monovisual_odometry"):
    src_dir = os.path.dirname(src_dir) 
if src_dir not in sys.path:
    sys.path.append(src_dir)

def parse_configs():
    parser = argparse.ArgumentParser()
    configs = edict(vars(parser.parse_args()))
    configs.root = src_dir
    configs.checkerboard = (4,6)
    configs.calib_dir = "dataset/calibration"
    configs.image_dir = "dataset/image/day/"
    configs.orig_size = [3024, 4032]
    configs.scale = 0.25
    return configs