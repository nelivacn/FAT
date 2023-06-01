import os
import sys
import json
import time
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


def get_feature_tester(fat, test_item_q, feat_q, gfbn):

    def get_feature_batch(fat, feat_q, inner_item_list):
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
                get_feature_batch(fat, feat_q, _item_list)
            LOGGER.info(
                f'get_feature_tester get None: {threading.currentThread().name}'
            )
            feat_q.put(None)
            break
        else:
            _item_list.append(test_item.split())
            if len(_item_list) == gfbn:
                get_feature_batch(fat, feat_q, _item_list)
                _item_list = []


if __name__ == '__main__':
    script = Path(__file__).stem
    pyfat_file = Path(sys.argv[1])
    fat_dir = pyfat_file.parent
    base_dir = os.getenv('NELIVA_CEPING_BASE_DIR')
    _dataset_dir = os.getenv('NELIVA_TEST_SET_DIR')
    if _dataset_dir in ['default', None]:
        dataset_dir = Path(r'/workspace/data/cluster')
    else:
        dataset_dir = Path(_dataset_dir)

    LOGGER.info(f'task_id is: {TASK_ID}')
    LOGGER.info(f'script: {script}')
    LOGGER.info(f'pyfat_file: {pyfat_file}')
    LOGGER.info(f'dataset_dir: {dataset_dir}')

    PROGRESS_FLAG = 'PROGRESS_'
    ERROR_FLAG = 'ERROR_'
    DONE_FLAG = 'DONE_'

    sample_file = dataset_dir / 'sample.txt'
    sample_count = 0
    with sample_file.open('r') as r_file:
        while line := r_file.readline():
            sample_count += 1
    sys.path.append(str(fat_dir))
    llp = os.environ['LD_LIBRARY_PATH']
    LOGGER.info(f'sys.path is: {sys.path}')
    LOGGER.info(f'LD_LIBRARY_PATH is: {llp}')
    LOGGER.info(f'sample count is: {sample_count}')

    LOGGER.info('test begin')

    try:
        from pyfat_implement import PyFAT

        fat = PyFAT(sample_count)
        LOGGER.info('fat init')
        assets_dir = str(fat_dir / 'assets')
        LOGGER.info(f'fat load {assets_dir}')
        LOGGER.info(f'device: {DEVICE}')
        fat.load(assets_dir, DEVICE)
        gfpn, gfbn = fat.get_feature_parallel_num()
        LOGGER.info(f'get_feature, P: {gfpn}, B: {gfbn}')
        res_feat_len = fat.get_feature_len()
        LOGGER.info(f'res feat len is: {res_feat_len}')

        load_sample_item_num = 9
        sample_item_q = Queue(1024)
        feat_q = Queue(1024)
        LOGGER.info('load sample')
        file2test_item_p_list = []
        for i in range(load_sample_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        sample_file, sample_item_q, i, load_sample_item_num)))
        for i in file2test_item_p_list:
            i.start()

        LOGGER.info('sample get_feature_item')
        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_sample_item_num, sample_count, sample_item_q, q_list))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(
                threading.Thread(
                    target=get_feature_tester, args=(fat, q_list[i], feat_q, gfbn)))
        for pp in p_list:
            LOGGER.info('in get_feature_tester')
            pp.start()

        LOGGER.info('feat insert begin')
        insert_count, feat_none_count = 0, 0
        while True:
            item = feat_q.get()
            if item is None:
                feat_none_count += 1
                if feat_none_count == gfpn:
                    if insert_count == sample_count:
                        LOGGER.info('sample insert success end')
                        MSG.info(f'{PROGRESS_FLAG}sample insert success')
                        break
                    else:
                        LOGGER.info(
                            f'insert count: {insert_count}\t sample_count: {sample_count}'
                        )
                        LOGGER.info('ERROR: sample insert count error')
                        MSG.info(
                            f'{PROGRESS_FLAG}ERROR: insert count: {insert_count}\t sample count: {sample_count}'
                        )
                        sys.exit(1)
            else:
                fat.insert_gallery(item[1], item[2], 0, item[0])
                insert_count += 1
                if insert_count % 200 == 0:
                    less_count = sample_count - insert_count
                    send_msg = json.dumps({
                        'total': sample_count,
                        'current': insert_count,
                        'currentPercent': float(insert_count) / sample_count
                    })
                    LOGGER.info(f'{PROGRESS_FLAG}{send_msg}')
                    MSG.info(f'{PROGRESS_FLAG}{send_msg}')

        LOGGER.info('insert gallery end')
        MSG.info(f'{PROGRESS_FLAG}insert gallery end')
        fat.unload_feature()
        LOGGER.info('unload_feature')
        MSG.info(f'{PROGRESS_FLAG}unload_feature')
        start_cluster = fat.start_cluster()
        LOGGER.info(f'start cluster res is: {start_cluster}')
        MSG.info(f'{PROGRESS_FLAG}start cluster res is: {start_cluster}')
        if not start_cluster:
            MSG.info(f'{ERROR_FLAG}start cluster res False')

        while progress_cluster := fat.query_progress_cluster() < 100:
            LOGGER.info(f'query_progress_cluster res is: {progress_cluster}')
            MSG.info(f'{PROGRESS_FLAG}query_progress_cluster res is: {progress_cluster}')
            time.sleep(66)

        LOGGER.info('cluster end')
        MSG.info(f'{PROGRESS_FLAG}cluster end')

        LOGGER.info('query start')
        query_count = 0
        all_clusters = fat.get_all_clusters()
        clusters_num = fat.get_clusters_num()
        assert len(all_clusters) == clusters_num
        for cluster_id in all_clusters:
            cluster_idx = fat.query_all_of_cluster(cluster_id)
            main_id = fat.query_cover_idx(cluster_id)
            cluster_size = fat.query_num_of_cluster(cluster_id)
            assert cluster_idx[0] == main_id
            assert cluster_size == len(cluster_idx)
            assert fat.query_cluster_res(cluster_idx[0]) == cluster_id
            assert fat.query_cluster_res(cluster_idx[-1]) == cluster_id
            query_count += 1
            if query_count % 50 == 0:
                LOGGER.info(f'cluster query: {query_count} all: {clusters_num}')

        item_query_count = 0
        with sample_file.open('r') as r_file:
            while line := r_file.readline():
                item_query_count += 1
                idx = int(line.split()[1])
                _ = fat.query_cluster_res(idx)
                if item_query_count % 5000 == 0:
                    LOGGER.info(f'item query: {item_query_count} all: {sample_count}')
        LOGGER.info('query end')
        MSG.info('query end')

        fat.unload_cluster()
        LOGGER.info('unload_cluster')
    except BaseException:
        traceback.print_exc()
        error = traceback.format_exc().splitlines()[-1]
        error = 'container_:' + error
        LOGGER.info(f'{ERROR_FLAG}{error}')
        MSG.info(f'{ERROR_FLAG}{error}')
