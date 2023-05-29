import os
import time
import logging
import logging.config
from queue import Queue
from typing import List
from pathlib import Path

LOG_LEVEL = int(os.getenv('LOG_LEVEL', 20))
BASE_DIR = os.getenv('CEPING_BASE_DIR', '/workspace/')
LOG_DIR = Path(BASE_DIR) / 'log'
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_now_str(format='%Y%m%d%H%M%S') -> str:
    return time.strftime(format)


def prepare_logging(name='eval'):
    TASK_LOG_DIR = LOG_DIR / name.split('_')[0]
    if not TASK_LOG_DIR.exists():
        TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(levelname)s - %(module)s - %(lineno)d: %(message)s'
            },
            'msg': {
                'format': '%(message)s'
            }
        },
        'handlers': {
            'stdout': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': LOG_LEVEL,
                'stream': 'ext://sys.stdout'
            },
            'msg': {
                'class': 'logging.StreamHandler',
                'formatter': 'msg',
                'level': LOG_LEVEL,
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'level': LOG_LEVEL,
                'filename': f'{TASK_LOG_DIR}/{name}.log'
            }
        },
        'loggers': {
            name: {
                'level': LOG_LEVEL,
                'handlers': ['stdout', 'file'],
                'propagate': False
            },
            'msg': {
                'level': LOG_LEVEL,
                'handlers': ['msg'],
                'propagate': False
            }
        }
    })


def file2q(file_name: Path, q: Queue, index: int, p_num: int):
    line_index = 0
    with file_name.open('r') as r_file:
        while line := r_file.readline():
            if line_index % p_num == index:
                q.put(line.strip())
            line_index += 1
        q.put(None)


def q2q_list(in_None_num: int, all_item_num: int, in_q: Queue, out_q_list: List[Queue]):
    index, len_out, none_count = 0, len(out_q_list), 0
    while True:
        item = in_q.get()
        if item is None:
            none_count += 1
            if none_count == in_None_num:
                if index == all_item_num:
                    for outp in out_q_list:
                        outp.put(None)
                else:
                    raise RuntimeError('q2q_list None num error')
                break
        else:
            out_q_list[index % len_out].put(item)
            index += 1
