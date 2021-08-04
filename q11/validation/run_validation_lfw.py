
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

featmap = {}

def get_feat(name, idx):
    global featmap
    key = (name, idx)
    if key in featmap:
        return featmap[key]
    img_path = 'lfw/%s/%s_%04d.jpg'%(name, name, idx)
    img = cv2.imread(img_path)
    ta = datetime.datetime.now()
    feat = x.get_feature(img)
    tb = datetime.datetime.now()
    if len(featmap)<10:
        print('cost:', (tb-ta).total_seconds())
    featmap[key] = feat
    return feat

positives = []
negatives = []
lines = open('lfw/pairs.txt', 'r').readlines()[1:]
print('total pairs:', len(lines))
pp = 0
for line in lines:
  pp += 1
  if pp%100==0:
      print('processing', pp)
  vec = line.strip().split()
  if len(vec)==4:
    n1, i1, n2, i2= vec
    i1 = int(i1)
    i2 = int(i2)
    feat1 = get_feat(n1, i1)
    feat2 = get_feat(n2, i2)
    issame = False
  else:
    n, i1, i2 = vec
    i1 = int(i1)
    i2 = int(i2)
    feat1 = get_feat(n, i1)
    feat2 = get_feat(n, i2)
    issame = True
  sim = x.get_sim(feat1, feat2)
  if issame:
      positives.append(sim)
  else:
      negatives.append(sim)

print(len(positives), len(negatives))

highest = [0.0, 0.0]
thresholds = np.arange(0, 1, 0.01)
for thresh in thresholds:
    pcorrect = len(np.where(positives>=thresh)[0])
    ncorrect = len(np.where(negatives<thresh)[0])
    acc = float(pcorrect+ncorrect) / (len(positives)+len(negatives))
    if acc>highest[0]:
        highest = [acc, thresh]
    #print(thresh, acc)

print('Highest:', highest)

