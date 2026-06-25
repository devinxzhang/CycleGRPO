import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, TypeVar, Generic
import os
import json
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image
import tqdm
from projects.vlm.vq_sam2.datasets.tfrecord_utils import read_tfrecord, read_tfrecord_files, decode_tf_batch
import time
import argparse


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='示例：解析命令行可选参数')
    # 添加可选参数
    parser.add_argument('--src', type=str, help='输入文件路径')
    parser.add_argument('--drc', type=str, help='输出文件路径')
    parser.add_argument('--info', type=str, help='sam info 文件路径')
    parser.add_argument('--nums', type=int, default=10000, help='数量，默认为10k')
    return parser.parse_args()

T = TypeVar('T')
R = TypeVar('R')


class ListProcessor(Generic[T, R]):
    """
    一个通用的列表处理器，使用多线程对列表进行切分并处理
    """

    def __init__(self, data: List[T], process_func: Callable[[T], R], num_threads: int = None):
        """
        初始化列表处理器

        Args:
            data: 需要处理的数据列表
            process_func: 处理单个元素的函数
            num_threads: 线程数量，默认为CPU核心数
        """
        self.data = data
        self.process_func = process_func
        self.num_threads = num_threads or threading.cpu_count()
        self.results = []
        self.lock = threading.Lock()

    def _process_chunk(self, chunk: List[T]) -> List[R]:
        """处理数据块的函数"""
        chunk_results = []
        for item in tqdm.tqdm(chunk):
            result = self.process_func(item)
            chunk_results.append(result)
        return chunk_results

    def _worker(self, chunk: List[T]) -> None:
        """工作线程函数"""
        chunk_results = self._process_chunk(chunk)
        # 使用锁来安全地更新共享结果列表
        with self.lock:
            self.results.extend(chunk_results)

    def process(self) -> List[R]:
        """
        开始多线程处理

        Returns:
            处理后的结果列表
        """
        # 清空之前的结果
        self.results = []

        # 计算每个线程要处理的数据量
        chunk_size = len(self.data) // self.num_threads
        if chunk_size == 0:
            chunk_size = 1

        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 切分数据并提交任务
            for i in range(0, len(self.data), chunk_size):
                chunk = self.data[i:i + chunk_size]
                executor.submit(self._worker, chunk)

        return self.results


def run(data, process_func, num_threads=32):
    # 多线程处理
    processor = ListProcessor(data, process_func, num_threads=num_threads)
    start_time = time.time()
    multi_thread_results = processor.process()
    multi_thread_time = time.time() - start_time
    print(f"多线程处理耗时: {multi_thread_time:.2f}秒")
    return multi_thread_results


def process_func(info_dict):
    image_name = info_dict['image_file']
    json_name = info_dict['json_file']
    image_path_src = os.path.join(src, image_name)
    image_path_drc = os.path.join(drc, image_name)
    json_path_src = os.path.join(src, json_name)
    json_path_drc = os.path.join(drc, json_name)
    os.system(f"cp {image_path_src} {image_path_drc}")
    os.system(f"cp {json_path_src} {json_path_drc}")
    return True

def process_tfrecord(path):
    dataset, count = read_tfrecord(path)
    ret = []
    for batch in dataset:
        ret.append(batch)
    return ret

if __name__ == "__main__":
    args = parse_arguments()
    src = args.src
    drc = args.drc
    info = args.info
    nums = args.nums
    with open(info, "r") as f:
        datas = json.load(f)
    datas = datas[:nums]
    with open(os.path.join(drc, "../sam_infos.json"), "w") as f:
        json.dump(datas, f)
    process_func(datas[0])
    processor = ListProcessor(data=datas, process_func=process_func, num_threads=100)
    results = processor.process()