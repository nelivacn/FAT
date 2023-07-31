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


def get_res(fat, test_item_q, res_q, bn, isDet=False):

    def get_res_batch(fat, _q, inner_item_list, _isDet=False):
        _img_data_list = []
        _pt1_list, _pt2_list = [], []
        for _i in inner_item_list:
            _img_data = cv2.imread(_i[1], cv2.IMREAD_COLOR)
            _img_data_list.append(_img_data)
            _pt1_list.append((int(_i[2]), int(_i[3])))
            _pt2_list.append((int(_i[4]), int(_i[5])))
        if _isDet:
            _res = fat.get_vehicle_bbox(_img_data_list)
        else:
            _res = fat.get_vehicle_info(_img_data_list, _pt1_list, _pt2_list)
        for index in range(len(_img_data_list)):
            _resi = _res[index]
            _test_itemi = inner_item_list[index]
            reitem = [_test_itemi[0], _resi]
            _q.put(reitem)

    _item_list = []
    while True:
        test_item = test_item_q.get()
        if test_item is None:
            if len(_item_list) > 0:
                get_res_batch(fat, res_q, _item_list, isDet)
            LOGGER.info(
                f'get None: {threading.currentThread().name}'
            )
            res_q.put(None)
            break
        else:
            _item_list.append(test_item.split())
            if len(_item_list) == bn:
                get_res_batch(fat, res_q, _item_list, isDet)
                _item_list = []


if __name__ == '__main__':
    script = Path(__file__).stem
    pyfat_file = Path(sys.argv[1])
    fat_dir = pyfat_file.parent
    base_dir = os.getenv('NELIVA_CEPING_BASE_DIR')
    _dataset_dir = os.getenv('NELIVA_TEST_SET_DIR')
    if _dataset_dir in ['default', None]:
        dataset_dir = Path(r'/workspace/data/vehicle')
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

        fat = PyFAT()
        LOGGER.info('fat init')
        assets_dir = str(fat_dir / 'assets')
        LOGGER.info(f'fat load {assets_dir}')
        # LOGGER.info(f'device: {DEVICE}')
        # fat.load(assets_dir, DEVICE)
        fat.load(assets_dir)
        gipn, gibn = fat.get_info_parallel_num()
        gdpn, gdbn = fat.get_detect_parallel_num()
        LOGGER.info(
            f'get_info, P: {gipn}, B: {gibn}, get_det, P: {gdpn}, B: {gdbn}'
        )

        load_sample_item_num = 9
        sample_item_q = Queue(1024)
        info_q = Queue(1024)
        det_q = Queue(1024)
        LOGGER.info('get info test load sample')
        file2test_item_p_list = []
        for i in range(load_sample_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        sample_file, sample_item_q, i, load_sample_item_num)))
        for i in file2test_item_p_list:
            i.start()

        LOGGER.info('sample get_info')
        q_list = [Queue(1024) for _ in range(gipn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_sample_item_num, sample_count, sample_item_q, q_list))
        q2q_list_p.start()

        p_list = []
        for i in range(gipn):
            p_list.append(
                threading.Thread(
                    target=get_res, args=(fat, q_list[i], info_q, gibn)))
        for pp in p_list:
            LOGGER.info('in get_info')
            pp.start()

        info_count, info_none_count = 0, 0
        while True:
            item = info_q.get()
            if item is None:
                info_none_count += 1
                if info_none_count == gipn:
                    if info_count == sample_count:
                        LOGGER.info('sample get info success end')
                        MSG.info(f'{PROGRESS_FLAG}sample insert success')
                        break
                    else:
                        LOGGER.info(
                            f'get info count: {info_count}\t sample_count: {sample_count}'
                        )
                        LOGGER.info('ERROR: sample get info count error')
                        MSG.info(
                            f'{PROGRESS_FLAG}ERROR: get info count: {info_count}\t sample count: {sample_count}'
                        )
                        sys.exit(1)
            else:
                LOGGER.info(item)
                info_dict = json.loads(item[1])
                LOGGER.info(info_dict['plate_pos_pt1'])
                LOGGER.info(info_dict['plate_pos_pt2'])
                LOGGER.info(info_dict['plate_color'])
                LOGGER.info(info_dict['plate_num'])
                info_count += 1
                if info_count % 500 == 0:
                    less_count = sample_count - info_count
                    send_msg = json.dumps({
                        'total': sample_count,
                        'current': info_count,
                        'currentPercent': float(info_count) / sample_count
                    })
                    LOGGER.info(f'{PROGRESS_FLAG}{send_msg}')
                    MSG.info(f'{PROGRESS_FLAG}{send_msg}')

        LOGGER.info('get det test load sample')
        file2test_item_p_list = []
        for i in range(load_sample_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        sample_file, sample_item_q, i, load_sample_item_num)))
        for i in file2test_item_p_list:
            i.start()

        LOGGER.info('sample get_det')
        q_list = [Queue(1024) for _ in range(gdpn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_sample_item_num, sample_count, sample_item_q, q_list))
        q2q_list_p.start()

        p_list = []
        for i in range(gdpn):
            p_list.append(
                threading.Thread(
                    target=get_res, args=(fat, q_list[i], det_q, gdbn, True)))
        for pp in p_list:
            LOGGER.info('in get_det')
            pp.start()

        det_count, det_none_count = 0, 0
        while True:
            item = det_q.get()
            if item is None:
                det_none_count += 1
                if det_none_count == gdpn:
                    if det_count == sample_count:
                        LOGGER.info('sample get info success end')
                        MSG.info(f'{PROGRESS_FLAG}sample insert success')
                        break
                    else:
                        LOGGER.info(
                            f'get det count: {det_count}\t sample_count: {sample_count}'
                        )
                        LOGGER.info('ERROR: sample get det count error')
                        MSG.info(
                            f'{PROGRESS_FLAG}ERROR: get det count: {det_count}\t sample count: {sample_count}'
                        )
                        sys.exit(1)
            else:
                LOGGER.info(item)
                det_list = json.loads(item[1])
                for det_dict in det_list:
                    LOGGER.info(det_dict['car_pos_pt1']['x'])
                    LOGGER.info(det_dict['car_pos_pt1']['y'])
                    LOGGER.info(det_dict['car_pos_pt2']['x'])
                    LOGGER.info(det_dict['car_pos_pt2']['y'])
                det_count += 1
                if det_count % 500 == 0:
                    less_count = sample_count - det_count
                    send_msg = json.dumps({
                        'total': sample_count,
                        'current': det_count,
                        'currentPercent': float(det_count) / sample_count
                    })
                    LOGGER.info(f'{PROGRESS_FLAG}{send_msg}')
                    MSG.info(f'{PROGRESS_FLAG}{send_msg}')

    except BaseException:
        traceback.print_exc()
        error = traceback.format_exc().splitlines()[-1]
        error = 'container_:' + error
        LOGGER.info(f'{ERROR_FLAG}{error}')
        MSG.info(f'{ERROR_FLAG}{error}')
