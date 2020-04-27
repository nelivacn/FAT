
import sys
import os
import datetime
import numpy as np
import cv2

import argparse

parser = argparse.ArgumentParser(description='Run FAT online validation.')
parser.add_argument('path', type=str, default='../cxxexample',
                    help='fat implementation path')

args = parser.parse_args()


fat_path = args.path
sys.path.append(fat_path)
from pyfat_implement import PyFAT

x = PyFAT()
assets_path = os.path.join(fat_path, 'assets')
x.load(assets_path)
feat_len = x.get_feature_length()
print('feat length:', feat_len)
img = cv2.imread('../../files/test-images/0.jpg')
feat1 = x.get_feature(img, 0)
#feat2 = x.get_feature(img, 0)
#sim = x.get_sim(feat1, 0, feat2, 0)
#print('sim:', sim)
for _ in range(10):
    ta = datetime.datetime.now()
    feat = x.get_feature(img)
    tb = datetime.datetime.now()
    print('cost:', (tb-ta).total_seconds())
    assert len(feat)==feat_len

