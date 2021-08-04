import argparse
import cv2
import time
import json
import sys
import numpy as np
import pickle
import os
import glob
import mxnet as mx
from numpy.linalg import norm as l2norm
import datetime
from skimage import transform as trans
import face_detection

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size = 112, mode='arcface'):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    #lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    tform.estimate(lmk, arcface_src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped

class PyFAT:

    def __init__(self):
        self.ctx = mx.cpu()
        self.image_size = (112,112)
        self.det_size = 224
        self.do_flip = False


    def load(self, rdir):
        self.detector = face_detection.get_retinaface(os.path.join(rdir, 'det', 'R50-0000.params'))
        self.detector.prepare(ctx_id=-1)
        rec_dir = os.path.join(rdir, 'rec')
        _files = os.listdir(rec_dir)
        param_files = []
        for _file in _files:
            if _file.endswith('.params'):
                param_files.append(_file)
        param_file = sorted(param_files)[-1]
        prefix = os.path.join(rec_dir, param_file[:-12])
        epoch = int(param_file[-11:-7])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = (112, 112)
        model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
        #model = mx.mod.Module(symbol=sym, context=ctx)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, self.image_size[0], self.image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.rec_model = model

    def get_feature_length(self):
        return 512

    def detect(self, img):
        #print(img.shape)
        S = self.det_size
        if img.shape[1]>img.shape[0]:
            det_scale = float(S) / img.shape[1]
            width = S
            height = float(img.shape[0]) / img.shape[1] * S
            height = int(height)
        else:
            det_scale = float(S) / img.shape[0]
            height = S
            width = float(img.shape[1]) / img.shape[0] * S
            width = int(width)
        img_resize = cv2.resize(img, (width, height))
        img_det = np.zeros( (S,S,3), dtype=np.uint8)
        img_det[:height,:width,:] = img_resize
        bboxes, det_landmarks = self.detector.detect(img_det, threshold=0.5)
        bboxes /= det_scale
        det_landmarks /= det_scale
        return bboxes, det_landmarks

    def im2mxdb(self, img, doflip=False):
        if not doflip:
            input_blob = np.zeros( (1, 3, img.shape[0], img.shape[1]),dtype=np.float32)
        else:
            input_blob = np.zeros( (2, 3, img.shape[0], img.shape[1]),dtype=np.float32)
        rimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rimg = np.transpose(rimg, (2,0,1)) #3*112*112, RGB
        input_blob[0] = rimg
        if doflip:
            input_blob[1] = rimg[:,:,::-1]
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        return db

    def get_feature(self, img, im_type=0):
        bboxes, det_landmarks = self.detect(img)
        if bboxes.shape[0]==0:
            return np.zeros( (self.get_feature_length(),), dtype=np.float32 )
        det = bboxes
        area = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        box_cw = (det[:,2]+det[:,0]) / 2
        box_ch = (det[:,3]+det[:,1]) / 2
        dist_cw = box_cw - img.shape[1]/2
        dist_ch = box_ch - img.shape[0]/2
        score = area - (dist_cw**2 + dist_ch**2)*2.0
        bindex = np.argmax(score)
        bbox = bboxes[bindex]
        det_landmark = det_landmarks[bindex]
        aimg = norm_crop(img, det_landmark)
        db112 = self.im2mxdb(aimg, self.do_flip)
        self.rec_model.forward(db112, is_train=False)
        feat = self.rec_model.get_outputs()[-1].asnumpy()
        feat = np.mean(feat, axis=0)
        feat /= l2norm(feat)
        return feat

    def get_quality(self, img, im_type=0):
        return np.random.random()

    def get_sim(self, feat1, feat2):
        return np.dot(feat1, feat2)

