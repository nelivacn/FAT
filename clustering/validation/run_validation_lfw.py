
import sys
import os
import datetime
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Run FAT online validation.')
parser.add_argument('path', type=str, default='../pyexample',
                    help='fat implementation path')
args = parser.parse_args()
fat_path = args.path

sys.path.append(fat_path)
from pyfat_implement import PyFAT


#N = 7701
N = 1000 #for test purpose

x = PyFAT(N=N)
assets_path = os.path.join(fat_path, 'assets')
x.load(assets_path)
#feat_len = x.get_feature_length()
#print('feat length:', feat_len)

images = {}
lines = open('lfw/pairs.txt', 'r').readlines()[1:]
print('total pairs:', len(lines))
for line in lines:
  vec = line.strip().split()
  if len(vec)==4:
    n1, i1, n2, i2= vec
    i1 = int(i1)
    i2 = int(i2)
    images[(n1, i1)] = 1
    images[(n2,i2)] = 1
  else:
    n, i1, i2 = vec
    i1 = int(i1)
    i2 = int(i2)
    images[(n,i1)] = 1
    images[(n,i2)] = 1

print('total-images:', len(images))
saved_feats = []

for key in images:
    if len(saved_feats)%100==0:
        print('processing', len(saved_feats))
    name, idx = key
    img_path = 'lfw/%s/%s_%04d.jpg'%(name, name, idx)
    img = cv2.imread(img_path)
    feat = x.get_feature(img)
    saved_feats.append(feat)
    if len(saved_feats)==N:
        break

for i, feat in enumerate(saved_feats):
    x.insert(feat, i)

x.cluster()
x.gen_res('output.txt')

