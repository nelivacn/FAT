import argparse
import cv2
import sys
import numpy as np
import os
from numpy.linalg import norm as l2norm

class PyFAT:

    def __init__(self, N, K):
        self.image_size = (128, 256)
        self.N = N
        self.K = K
        self.G = np.zeros((N, self.get_feature_length()), dtype=np.float32)


    def load(self, rdir):
        self.reid_model = None #provide your own model

    def get_feature_length(self):
        return 512


    def get_feature(self, img, im_type=0):
        #feat = np.random.random((self.get_feature_length(),), dtype=np.float32) #random output
        feat = self.reid_model.predict(img)
        assert len(feat)==self.get_feature_length()
        feat /= l2norm(feat)
        return feat

    def insert_gallery(self, feat, idx):
        self.G[idx,:] = feat

    def finalize(self):
        pass

    def get_topk(self, query_feat):
        sims = np.dot(self.G, query_feat)
        sims = (sims+1) / 2 #to [0,1]
        ret_idxs = np.argsort(sims)[::-1][:self.K]
        ret_sims = sims[ret_idxs]
        return ret_idxs, ret_sims

