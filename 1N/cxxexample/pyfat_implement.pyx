# cython: language_level=3
import numpy as np
# from threading import Lock, Thread
from multiprocessing import Queue, Process, Lock
# from multiprocessing.managers import BaseManager
# from queue import Queue
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
import time


# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "<mutex>" namespace "std" nogil:
    cdef cppclass recursive_mutex:
        pass
    cdef cppclass lock_guard[T]:
        lock_guard(recursive_mutex mm)


cdef extern from "source/fat_core.h" namespace "FAT":
    cdef struct Image:
        int nC
        int nW
        int nH
        char* pData

    cdef cppclass FATImpl:
        FATImpl()
        void Load(string rdir)
        int FeatureLength()
        void GetFeature(char* img, int im_width, int im_height, int im_channel, int im_type, float* feat)
        void GetFeatures(const vector[Image] &vImage, vector[float*] &vFeatureStream)
    
    cdef cppclass FATGalleryQuery:
        FATGalleryQuery()
        void Finalize(float* gallery_feats, int* gallery_labels, int N, int K)
        void GetTopK(float* im_feat, int* topk, float* sim)
        void GetTopKs(const vector[float*] &im_feats, vector[int*] &topks, vector[float*] &sims);




cdef class PyFAT:
    # 人脸处理句柄 
    cdef FATImpl face_handler
    # 特征库句柄 
    cdef FATGalleryQuery db_handler
    # topk 
    cdef int K
    # 底库大小 
    cdef int N
    # 特征维度 
    cdef int dim
    # 检索最大并发 
    cdef int parallel_query
    # 特征提取最大并发 
    cdef int parallel_ext
    # 显卡ID
    cdef int device_id

    # idx字典
    cdef object idx_buffer
    # 特征缓存
    cdef object feature_buffer
    # idx累加器
    cdef object idxcnt

    # 特征提取请求队列
    cdef object ext_queue_in
    # 特征提取响应队列的列表
    cdef object ext_queue_out
    # 特征检索请求队列
    cdef object query_queue_in
    # 特征检索响应队列的列表
    cdef object query_queue_out
    # 特征插入请求队列
    cdef object insert_queue_in
    # 特征插入响应队列的列表
    cdef object insert_queue_out
    # 特征冻结请求队列
    cdef object finalize_queue_in
    # 特征冻结响应队列
    cdef object finalize_queue_out

    # 特征提取的可用id列表
    cdef object ext_available_ids
    # 特征检索的可用id列表
    cdef object query_available_ids
    # 特征插入的可用id列表
    cdef object insert_available_ids

    # 特征提取的id锁
    cdef object ext_lk
    # 特征检索的id锁
    cdef object query_lk
    # 特征插入的id锁
    cdef object insert_lk

    # 特征提取的进程
    cdef object ext_process
    # 特征检索(插入/冻结)的进程
    cdef object query_process
    # 特征提取进程初始化状态队列
    cdef object ext_init_status_queue
    # 特征检索进程初始化状态队列
    cdef object query_init_status_queue

    #N is number of gallery images, K is top K
    def __cinit__(self, N, K, device_id = 0, dim = 512, parallel_ext = 4, parallel_query = 16):
        self.N = N
        self.K = K
        self.dim = dim
        self.device_id = device_id
        self.parallel_ext = parallel_ext
        self.parallel_query = parallel_query
        self.face_handler = FATImpl()
        self.db_handler = FATGalleryQuery()

        self.finalize_queue_in = Queue()
        self.finalize_queue_out = Queue()

        self.insert_queue_in = Queue()
        self.insert_queue_out = [Queue() for v in range(parallel_ext)] 
        self.insert_available_ids = [True for v in range(parallel_ext)]
        self.insert_lk = Lock()

        self.ext_queue_in = Queue()
        self.ext_queue_out = [Queue() for v in range(parallel_ext)]
        self.ext_available_ids = [True for i in range(parallel_ext)]
        self.ext_lk = Lock()

        self.query_queue_in = Queue()
        self.query_queue_out = [Queue() for v in range(parallel_query) ]
        self.query_available_ids = [True for i in range(parallel_query)]
        self.query_lk = Lock()

        self.ext_init_status_queue = Queue()
        self.query_init_status_queue = Queue()
        self.parallel_ext = parallel_ext
        self.parallel_query = parallel_query

        self.idxcnt = 0

    def get_parallel_num(self):
        return self.parallel_ext, self.parallel_query

    def load(self, rdir):
        self.ext_process = Process(target = self._ext, args=(rdir,))
        self.query_process = Process(target = self._query, args=(rdir,))
        self.ext_process.daemon = True
        self.query_process.daemon = True
        self.ext_process.start()
        self.query_process.start()
        status = self.ext_init_status_queue.get()
        if status  != 0:
            raise Exception(f"load error : {status}")
        status = self.query_init_status_queue.get()
        if status  != 0:
            raise Exception(f"load error : {status}")

    def insert_gallery(self, np.ndarray [float, ndim=1, mode="c"] feat not None, int idx, int label):
        while True:
            cidx = self._get_available_id(self.insert_lk, self.insert_available_ids)
            if cidx is None:
                time.sleep(0.001)
                continue
            break
        self.insert_queue_in.put((feat, idx, label, cidx))
        _ = self.insert_queue_out[cidx].get()
        self._return_available_id(self.insert_lk, self.insert_available_ids, cidx)

    def finalize(self):
        self.finalize_queue_in.put(True)
        _ = self.finalize_queue_out.get()

    def get_feature(self, np.ndarray[char, ndim=3, mode = "c"] img not None):
        while True:
            idx = self._get_available_id(self.ext_lk, self.ext_available_ids)
            if idx is None:
                time.sleep(0.001)
                continue
            break
        self.ext_queue_in.put((img, idx))
        feature = self.ext_queue_out[idx].get()
        self._return_available_id(self.ext_lk, self.ext_available_ids, idx)
        return feature

    def get_topk(self, np.ndarray[float, ndim=1, mode = "c"] feat not None):
        while True:
            idx = self._get_available_id(self.query_lk, self.query_available_ids)
            if idx is None:
                time.sleep(0.001)
                continue
            break
        self.query_queue_in.put((feat, idx))
        topk,sim = self.query_queue_out[idx].get()
        self._return_available_id(self.query_lk, self.query_available_ids, idx)
        return topk, sim

    def _get_multi_feature(self, img_list):
        cdef vector[Image] images
        cdef vector[float*] features
        feat_list = []

        size = len(img_list)
        images.resize(size)
        features.resize(size)
        for i,img in enumerate(img_list):
            images[i].nH, images[i].nW, images[i].nC = (<object> img).shape
            assert images[i].nC == 3 
            images[i].pData = <char*> np.PyArray_DATA(img)
            feat = np.zeros( (self.dim, ), dtype=np.float32 )
            features[i] = <float*> np.PyArray_DATA(feat)
            feat_list.append(feat)

            
        self.face_handler.GetFeatures(images, features)
        return feat_list


    def _get_multi_topk(self, feat_list):
        cdef vector[float*] features
        cdef vector[int*] topks
        cdef vector[float*] sims

        size = len(feat_list)
        features.resize(size)
        topks.resize(size)
        sims.resize(size)

        topk_list = []
        sim_list = []
        for i in range(size):
            features[i] = <float*> np.PyArray_DATA(feat_list[i])
            topk = np.zeros( (self.K, ), dtype=np.int32)
            sim = np.zeros( (self.K, ), dtype=np.float32)
            sims[i] = <float*> np.PyArray_DATA(sim)
            topks[i] = <int*> np.PyArray_DATA(topk)
            topk_list.append(topk)
            sim_list.append(sim)
        self.db_handler.GetTopKs(features, topks, sims)
        return topk_list, sim_list


    def _insert(self, feat, idx):
        self.idx_buffer[self.idxcnt] = idx
        self.feature_buffer[self.idxcnt,:] = feat
        # print(self.feature_buffer[self.idxcnt,0:5], self.idx_buffer[self.idxcnt])
        # print(feat[0:5], idx)
        self.idxcnt += 1

    def _finalize(self):
        # status = self.db_handler.Insert(<float*>np.PyArray_DATA(self.feature_buffer), self.dim, self.N)
        self.db_handler.Finalize(
            <float*> np.PyArray_DATA(np.asarray(self.feature_buffer)),
            <int*> np.PyArray_DATA(np.asarray(self.idx_buffer)), 
            self.N, self.K)

    def _ext(self, rdir):
        self.face_handler.Load(bytes(rdir, encoding='utf-8'))
        # status = self.face_handler.Init(self.device_id, self.parallel_ext, rdir.encode('utf-8'))
        self.ext_init_status_queue.put(0) 
        while True:
            # process ext message
            size = self.ext_queue_in.qsize()
            prefetch = min(size, self.parallel_ext)
            if prefetch == 0:
                time.sleep(0.001)
                continue
            images = [] 
            idxs = []
            for i in range(prefetch):
                image, idx = self.ext_queue_in.get()
                images.append(image)
                idxs.append(idx)
            feature_arr = self._get_multi_feature(images)
            for i in range(prefetch):
                self.ext_queue_out[idxs[i]].put(feature_arr[i])

    def _query(self, rdir):
        self.idx_buffer = np.zeros( (self.N,), dtype=np.int32 )
        self.feature_buffer = np.zeros( (self.N, self.dim), dtype=np.float32 )
        
        # status = self.db_handler.Init(self.device_id, self.dim, self.N, self.parallel_query, rdir.encode('utf-8'))
        self.query_init_status_queue.put(0)
        while True:

            # process finalize message
            final_size = self.finalize_queue_in.qsize()
            if final_size > 0:
                _ = self.finalize_queue_in.get()
                self._finalize()
                self.finalize_queue_out.put(True)
                continue

            # process insert message
            insert_size = self.insert_queue_in.qsize()
            if insert_size > 0:
                for i in range(insert_size):
                    fea,plabel,flabel,idx = self.insert_queue_in.get()
                    self._insert(fea, plabel)
                    self.insert_queue_out[idx].put(True)
                continue

            # process query message
            size = self.query_queue_in.qsize()
            prefetch = min(size, self.parallel_query)
            if prefetch == 0:
                continue
            features = [] 
            idxs = []
            for i in range(prefetch):
                feature, idx = self.query_queue_in.get()
                features.append(feature)
                idxs.append(idx)
            
            topk, sim = self._get_multi_topk(features)
            for i in range(prefetch):
                topk[i] = self._conv_topk(topk[i])
                sim[i] = self._conv_sim(sim[i])
                self.query_queue_out[idxs[i]].put((topk[i],sim[i]))

    def _get_available_id(self, lk, ids):
        lk.acquire()
        for i,v in enumerate(ids):
            if v:
                ids[i] = False
                lk.release()
                return i
        lk.release()
        return None

    def _return_available_id(self, lk, ids, idx):
        lk.acquire()
        ids[idx] = True
        lk.release()
        return None

    def _conv_topk(self,topk):
        output = []
        for v in topk.tolist():
            output.append(self.idx_buffer[v])
        return np.require(output,dtype=np.int32)

    def _conv_sim(self,sim):
        output = []
        for v in sim.tolist():
            output.append((v+1)/2)
        return np.require(output,dtype=np.float32)

