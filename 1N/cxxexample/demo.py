from cProfile import label
from queue import Queue
import threading
import time
import cv2

from pyfat_implement import PyFAT

db_size = 10
query_size = 20

class Loader(object):

    def __init__(self, db_size) -> None:
        super().__init__()
        self.db_size = db_size
    

    def load_data(self, q_list):
        '''
        加载测试数据
        放到队列列表中
        '''
        count = 0
        while count < self.db_size:
            for q_i in q_list:
                if count == self.db_size:
                    break
                img = cv2.imread("images/N/t3.jpeg")
                q_i.put( (img, count+1, count+1) )
                time.sleep(0.001)
                count += 1
        for q_i in q_list:
            q_i.put(None)
        print(threading.currentThread( ), "*******************get out loader******************")


def get_feature(fat, in_q, out_q):
    '''
    从队列in_q读数据，提取特征后将相关数据和特征放入队列out_q
    '''
    while True:
        item = in_q.get()
        if item is None:
            break
        img, idx, label = item
        feat = fat.get_feature(img)
        out_q.put( (feat, idx, label) )
    out_q.put(None)
    print(threading.currentThread( ), "******************get out get_feature******************")


def item_distribute(q, q_list, stop_count):
    '''
    将一个队列内的元素分配给另一个队列列表
    '''
    while True:
        for _q in q_list:
            item = q.get()
            if item is None:
                stop_count -= 1
            if stop_count == 0:
                break
            _q.put(item)
        if stop_count == 0:
            break
    for _q in q_list:
        _q.put(None)
    print(threading.currentThread( ), "******************get out item_distribute******************")


def get_topk(fat, in_q, out_q):
    '''
    从队列in_q读数据，获得检索结果后将相关数据和结果放入队列out_q
    '''
    while True:
        feat = in_q.get()
        if feat is None:
            break
        topk, sim = fat.get_topk(feat[0])
        out_q.put((topk, sim))
    out_q.put(None)
    print(threading.currentThread( ), "******************get out get_topk******************")


def main():
    fat = PyFAT(db_size, 1)
    fat.load('./assets')
    fpnum, tpnum = fat.get_parallel_num()
    # 底库测试数据队列列表
    db_data_Q_list = [Queue(5) for _ in range(fpnum)]
    # 底库特征队列
    db_feat_Q = Queue(50)

    loader = Loader(db_size)
    # 启动线程向队列内填充测试数据（底库）
    db_load_data_t = threading.Thread(
        target=loader.load_data, args=(db_data_Q_list,)
    )
    db_load_data_t.start()
    # 并发提特征，放入底库特征队列中
    get_f_list = [
        threading.Thread(
            target=get_feature, args=(fat, db_data_Q_list[i], db_feat_Q)
        ) for i in range(fpnum)
    ]
    print("start get feature threads...")
    for get_f_t in get_f_list:
        get_f_t.start()
    # # 建库
    print("insert gallery...")
    stop_count = fpnum
    while item := db_feat_Q.get():
        if item is None:
            stop_count -= 1
            if stop_count == 0:
                break
            continue
        fat.insert_gallery(*item)
    print("finalize")
    fat.finalize()
    # # 探测测试数据队列列表
    det_data_Q_list = [Queue(10) for _ in range(fpnum)]
    # # 探测特征队列列表
    feat_Q_list = [Queue(10) for _ in range(tpnum)]
    # # 探测特征队列，结果队列
    det_feat_Q, res_Q = Queue(50), Queue(50)

    # # 队列内填充测试数据（探测）
    print("start det img loader")
    det_loader = Loader(query_size)
    det_load_data_t = threading.Thread(
        target=det_loader.load_data, args=(det_data_Q_list,)
    )
    det_load_data_t.start()
    # 并发提特征
    print("start det img extract")
    get_f_list = [
        threading.Thread(
            target=get_feature, args=(fat, det_data_Q_list[i], det_feat_Q)
        ) for i in range(fpnum)
    ]
    for get_f_t in get_f_list:
        get_f_t.start()
    # 特征分发给每个检索队列
    dist_t = threading.Thread(
        target=item_distribute, args=(det_feat_Q, feat_Q_list, fpnum)
    )
    dist_t.start()
    # 并发检索，所有检索结果在队列res_Q内
    print("start det img query")
    get_t_list = [
        threading.Thread(
            target=get_topk, args=(fat, feat_Q_list[i], res_Q)
        ) for i in range(tpnum)
    ]
    for get_t_t in get_t_list:
        get_t_t.start()

    stop_count = tpnum
    
    while True:
        item = res_Q.get()
        print(item)
        if item is None:
            stop_count -= 1
            print("current stop count", stop_count)
        if stop_count == 0:
            break




if __name__ == '__main__':
    main()
