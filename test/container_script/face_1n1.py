import os
import sys
import json
import logging
import traceback
import threading
from queue import Queue
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent))

from ceping_util import file2q, q2q_list, get_now_str, prepare_logging

TASK_ID = os.getenv('NELIVA_TASK_ID')
if TASK_ID is None:
    sys.exit(9)
log_name = f'{TASK_ID}_{get_now_str()}'
prepare_logging(log_name)
LOGGER = logging.getLogger(log_name)
MSG = logging.getLogger('msg')
DEVICE = [0, 1]


def get_gallery_feature_tester(fat, test_item_q, feat_q, gfbn):

    def get_gallery_feature_batch(fat, feat_q, inner_item_list):
        _img_data_list = []
        for _i in inner_item_list:
            _img_data = cv2.imread(_i[2], cv2.IMREAD_COLOR)
            _img_data_list.append(_img_data)
        _suc, _feat = fat.get_feature(_img_data_list)
        for index in range(len(_img_data_list)):
            _suci = _suc[index]
            _feati = _feat[index]
            _test_itemi = inner_item_list[index]
            reitem = [_suci, _feati, int(_test_itemi[1]), int(_test_itemi[0])]
            feat_q.put(reitem)

    LOGGER.info('do get_feature_tester')
    _item_list = []
    while True:
        test_item = test_item_q.get()
        if test_item is None:
            if len(_item_list) > 0:
                get_gallery_feature_batch(fat, feat_q, _item_list)
            LOGGER.info(
                f'get_feature_tester get None: {threading.currentThread().name}'
            )
            feat_q.put(None)
            break
        else:
            _item_list.append(test_item.split())
            if len(_item_list) == gfbn:
                get_gallery_feature_batch(fat, feat_q, _item_list)
                _item_list = []


def get_probe_feature_tester(fat, test_item_q, feat_q, gfbn):

    def get_probe_feature_batch(fat, feat_q, inner_item_list):
        _img_data_list = []
        for _i in inner_item_list:
            _img_data = cv2.imread(_i[2], cv2.IMREAD_COLOR)
            _img_data_list.append(_img_data)
        _suc, _feat = fat.get_feature(_img_data_list)
        for index in range(len(_img_data_list)):
            _suci = _suc[index]
            _feati = _feat[index]
            _test_itemi = inner_item_list[index]
            reitem = [_suci, _feati, f'{_test_itemi[0]}_{_test_itemi[1]}']
            feat_q.put(reitem)

    LOGGER.info('do get_probe_feature_tester')
    _item_list = []
    while True:
        test_item = test_item_q.get()
        if test_item is None:
            LOGGER.info(
                f'get_probe_feature_tester get None: {threading.currentThread().name}'
            )
            if len(_item_list) > 0:
                get_probe_feature_batch(fat, feat_q, _item_list)
            feat_q.put(None)
            break
        else:
            _item_list.append(test_item.split())
            if len(_item_list) == gfbn:
                get_probe_feature_batch(fat, feat_q, _item_list)
                _item_list = []


def get_topk_tester(fat, feat_item_q, res_q, gtkbn):

    def get_topk_tester_batch(fat, _item_list_inner, res_q):
        _feat_data_list = []
        _issuc_list = []
        for _i in _item_list_inner:
            det_feat = _i[1]
            issuc = _i[0]
            _feat_data_list.append(det_feat)
            _issuc_list.append(issuc)
        idxs, sims = fat.get_topk(_feat_data_list, _issuc_list)
        for index in range(len(_feat_data_list)):
            _item_inner = _item_list_inner[index]
            _info = _item_inner[1]
            res_item = [_info, idxs[index][0], sims[index][0]]
            res_q.put(res_item)

    LOGGER.info('do get_topk_tester')
    _item_list = []
    while True:
        feat_item = feat_item_q.get()
        if feat_item is None:
            LOGGER.info(f'get_topk get None: {threading.currentThread().name}')
            if len(_item_list) > 0:
                get_topk_tester_batch(fat, _item_list, res_q)
            res_q.put(None)
            break
        else:
            _item_list.append(feat_item)
            if len(_item_list) == gtkbn:
                get_topk_tester_batch(fat, _item_list, res_q)
                _item_list = []


if __name__ == '__main__':
    script = Path(__file__).stem
    pyfat_file = Path(sys.argv[1])
    fat_dir = pyfat_file.parent
    base_dir = os.getenv('NELIVA_CEPING_BASE_DIR')
    _dataset_dir = os.getenv('NELIVA_TEST_SET_DIR')
    if _dataset_dir in ['default', None]:
        dataset_dir = Path(r'/workspace/data/face_1n1')
    else:
        dataset_dir = Path(_dataset_dir)

    LOGGER.info(f'task_id is: {TASK_ID}')
    LOGGER.info(f'script: {script}')
    LOGGER.info(f'pyfat_file: {pyfat_file}')
    LOGGER.info(f'dataset_dir: {dataset_dir}')

    PROGRESS_FLAG = 'PROGRESS_'
    ERROR_FLAG = 'ERROR_'
    DONE_FLAG = 'DONE_'

    gallery_file = dataset_dir / 'gallery.txt'
    probe_file = dataset_dir / 'probe.txt'
    gallery_count, probe_count = 0, 0
    with probe_file.open('r') as r_label:
        while line := r_label.readline():
            probe_count += 1
    with open(gallery_file, 'r') as r_label:
        while line := r_label.readline():
            gallery_count += 1
    sys.path.append(str(fat_dir))
    llp = os.environ['LD_LIBRARY_PATH']
    LOGGER.info(f'sys.path is: {sys.path}')
    LOGGER.info(f'LD_LIBRARY_PATH is: {llp}')
    LOGGER.info(f'gallery count is: {gallery_count}')
    LOGGER.info(f'probe count is: {probe_count}')

    LOGGER.info('test begin')

    try:
        from pyfat_implement import PyFAT

        fat = PyFAT(gallery_count, 1)
        LOGGER.info('fat init')
        assets_dir = str(fat_dir / 'assets')
        LOGGER.info(f'fat load {assets_dir}')
        LOGGER.info(f'device: {DEVICE}')
        fat.load(assets_dir, DEVICE)
        gfpn, gfbn = fat.get_feature_parallel_num()
        gtkpn, gtkbn = fat.get_topk_parallel_num()
        LOGGER.info(
            f'get_feature, P: {gfpn}, B: {gfbn}, get_topK, P: {gtkpn}, B: {gtkbn}'
        )

        load_test_item_num = 9
        gallery_test_item_q = Queue(1024)
        probe_test_item_q = Queue(1024)
        probe_feat_q = Queue(1024)
        LOGGER.info('load sample')
        file2test_item_p_list = []
        for i in range(load_test_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        gallery_file, gallery_test_item_q, i, load_test_item_num)))
        for i in file2test_item_p_list:
            i.start()

        LOGGER.info('gallery get_feature_item')
        gallery_feat_q = Queue(1024)
        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_test_item_num, gallery_count, gallery_test_item_q, q_list))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(
                threading.Thread(
                    target=get_gallery_feature_tester, args=(fat, q_list[i], gallery_feat_q, gfbn)))
        for pp in p_list:
            LOGGER.info('in get_feature_tester')
            pp.start()

        LOGGER.info('gallery insert begin')
        insert_count, gallery_feat_none_count = 0, 0
        while True:
            item = gallery_feat_q.get()
            if item is None:
                gallery_feat_none_count += 1
                if gallery_feat_none_count == gfpn:
                    if insert_count == gallery_count:
                        LOGGER.info('gallery insert success end')
                        MSG.info(f'{PROGRESS_FLAG}gallery insert success')
                        break
                    else:
                        LOGGER.info(
                            f'insert count: {insert_count}\t gallery_count: {gallery_count}'
                        )
                        LOGGER.info('ERROR: gallery insert count error')
                        MSG.info(
                            f'{ERROR_FLAG}ERROR: insert count: {insert_count}\t gallery count: {gallery_count}'
                        )
                        sys.exit(1)
            else:
                fat.insert_gallery(item[1], item[2], item[3], item[0])
                insert_count += 1
                if insert_count % 200 == 0:
                    less_count = gallery_count - insert_count
                    send_msg = json.dumps({
                        'total': gallery_count,
                        'current': insert_count,
                        'currentPercent': float(insert_count) / gallery_count
                    })
                    LOGGER.info(f'{PROGRESS_FLAG}{send_msg}')
                    MSG.info(f'{PROGRESS_FLAG}{send_msg}')

        LOGGER.info('finalize start')
        fat.finalize()
        LOGGER.info('finalize end')

        LOGGER.info('retrieval start')
        file2test_item_p_list = []
        for i in range(load_test_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(probe_file, probe_test_item_q, i, load_test_item_num)))
        for i in file2test_item_p_list:
            i.start()

        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_test_item_num, probe_count, probe_test_item_q, q_list))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(
                threading.Thread(
                    target=get_probe_feature_tester, args=(fat, q_list[i], probe_feat_q, gfbn)))
        for pp in p_list:
            LOGGER.info('in get_probe_feature_tester')
            pp.start()

        qq_list = [Queue(1024) for _ in range(gtkpn)]
        qq2q_list_p = threading.Thread(
            target=q2q_list, args=(gfpn, probe_count, probe_feat_q, qq_list))
        qq2q_list_p.start()

        res_q = Queue(1024)
        get_topk_p = []
        for i in range(gtkpn):
            get_topk_p.append(
                threading.Thread(
                    target=get_topk_tester, args=(fat, qq_list[i], res_q, gtkbn)))
        for ppp in get_topk_p:
            LOGGER.info('get_topk_tester start')
            ppp.start()

        det_gettopk_none_count, retrieval = 0, 0
        while True:
            item = res_q.get()
            if item is None:
                det_gettopk_none_count += 1
                if det_gettopk_none_count == gtkpn:
                    if retrieval == probe_count:
                        LOGGER.info('retrieval success end')
                        MSG.info(f'{DONE_FLAG} test success end')
                        break
                    else:
                        LOGGER.info(
                            f'retrieval count: {retrieval}\t probe_count: {probe_count}'
                        )
                        LOGGER.info('ERROR: retrieval error')
                        MSG.info(
                            f'{ERROR_FLAG}retrieval count: {retrieval}\t probe count: {probe_count}'
                        )
                        sys.exit(1)
            else:
                res_str = ' '.join([str(i) for i in item])
                LOGGER.info(res_str)
                retrieval += 1
                if retrieval % 200 == 0:
                    send_msg = json.dumps({
                        'all': probe_count,
                        'num': retrieval,
                        'percent': float(retrieval) / probe_count
                    })
                    LOGGER.info(f'{PROGRESS_FLAG}{send_msg}')
                    MSG.info(f'{PROGRESS_FLAG}{send_msg}')

    except BaseException:
        traceback.print_exc()
        error = traceback.format_exc().splitlines()[-1]
        error = 'container_:' + error
        LOGGER.info(f'{ERROR_FLAG}{error}')
        MSG.info(f'{ERROR_FLAG}{error}')
