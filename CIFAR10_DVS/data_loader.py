from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from typing import Callable, Dict, Optional, Tuple
import numpy as np

from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
# import np_savez
#
# def read_aedat_save_to_np(bin_file: str, np_file: str):
#     events = CIFAR10DVS.load_origin_data(bin_file)
#     np_savez(np_file,
#              t=events['t'],
#              x=events['x'],
#              y=events['y'],
#              p=events['p']
#              )
#     print(f'Save [{bin_file}] to [{np_file}].')

def main(extract_root: str, events_np_root: str):
    '''
    :param extract_root: Root directory path which saves extracted files from downloaded files
    :type extract_root: str
    :param events_np_root: Root directory path which saves events files in the ``npz`` format
    :type events_np_root:
    :return: None

    This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
    '''
    for class_name in os.listdir(extract_root):
        aedat_dir = os.path.join(extract_root, class_name)
        np_dir = os.path.join(events_np_root, class_name)
        for bin_file in os.listdir(aedat_dir):
            source_file = os.path.join(aedat_dir, bin_file)
            target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
            print(f'Start to convert [{source_file}] to [{target_file}].')
            # tpe.submit(CIFAR10DVS.read_aedat_save_to_np, source_file,
            #            target_file)
            CIFAR10DVS.read_aedat_save_to_np(source_file, target_file)
if __name__ == '__main__':
    extract_root = './data/extract'
    events_np_root = './data/events_np'
    main(extract_root, events_np_root)