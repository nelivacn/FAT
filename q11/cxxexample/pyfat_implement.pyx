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
        float GetSim(float* im1_feat, float* im2_feat)

cdef class PyFAT:

    cdef FATImpl obj
    cdef int feat_length

    def __cinit__(self):
        self.obj = FATImpl()
        self.feat_length = self.obj.FeatureLength()

    def get_feature_length(self):
        return self.feat_length

    def load(self, rdir):
        self.obj.Load(bytes(rdir, encoding='utf-8'))

    def get_feature(self, np.ndarray[char, ndim=3, mode = "c"] img not None, int im_type=0):
        feat = np.zeros( (self.feat_length, ), dtype=np.float32 )
        self.obj.GetFeature(<char*> np.PyArray_DATA(img), img.shape[1], img.shape[0], img.shape[2], im_type, <float*> np.PyArray_DATA(feat))
        return feat

    def get_quality(self, np.ndarray[char, ndim=3, mode = "c"] img not None, int im_type=0):
        return np.random.random()

    def get_sim(self, np.ndarray[float, ndim=1, mode = "c"] feat1 not None, np.ndarray[float, ndim=1, mode = "c"] feat2 not None):
        return self.obj.GetSim(<float*> np.PyArray_DATA(feat1), <float*> np.PyArray_DATA(feat2))

