import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "source/fat_core.h" namespace "FAT":
    cdef cppclass FATImpl:
        FATImpl()
        void Load(string rdir)
        int FeatureLength()
        void GetFeature(char* img, int im_width, int im_height, int im_channel, int im_type, float* feat)
        void Finalize(float* gallery_feats, int* gallery_labels, int N, int K)
        void GetTopK(float* im_feat, int* topk, float* sim)

cdef class PyFAT:

    cdef FATImpl obj
    cdef int feat_length
    cdef int K
    cdef np.float32_t [:, :] G_feats
    cdef long [:] G_labels

    #N is number of gallery images, K is top K
    def __cinit__(self, N, K):
        self.obj = FATImpl()
        self.feat_length = self.obj.FeatureLength()
        self.G_feats = np.zeros( (N, self.feat_length), dtype=np.float32 )
        self.G_labels = np.zeros( (N,), dtype=np.int64)
        self.K = K
        #self.idx = 0

    def get_feature_length(self):
        return self.feat_length

    def load(self, rdir):
        self.obj.Load(bytes(rdir, encoding='utf-8'))

    def insert_gallery(self, np.float32_t [:] feat not None, int idx, int label, int im_type=0):
        assert idx<self.G_feats.shape[0]
        self.G_feats[idx] = feat
        self.G_labels[idx] = label

    def finalize(self):
        self.obj.Finalize(<float*> np.PyArray_DATA(np.asarray(self.G_feats)), <int*> np.PyArray_DATA(np.asarray(self.G_labels)), self.G_feats.shape[0], self.K)


    def get_feature(self, np.ndarray[char, ndim=3, mode = "c"] img not None, int im_type=0):
        feat = np.zeros( (self.feat_length, ), dtype=np.float32 )
        self.obj.GetFeature(<char*> np.PyArray_DATA(img), img.shape[1], img.shape[0], img.shape[2], im_type, <float*> np.PyArray_DATA(feat))
        return feat

    def get_topk(self, np.ndarray[float, ndim=1, mode = "c"] feat not None):
        topk = np.zeros( (self.K, ), dtype=np.int)
        sim = np.zeros( (self.K, ), dtype=np.float32 )
        self.obj.GetTopK(<float*> np.PyArray_DATA(feat), <int*> np.PyArray_DATA(topk), <float*> np.PyArray_DATA(sim))
        return topk, sim

