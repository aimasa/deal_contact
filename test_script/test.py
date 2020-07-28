import progressbar
import time
# 先定义一个进度条
# http://blog.useasp.net/

# pbar = progressbar.ProgressBar(maxval=100,widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage()]).start()
# for i in range(100):
#     # 更新进度条
#     time.sleep(0.1)
#     pbar.update(i + 1)
#
# pbar.finish()
import re
#encoding=utf-8
import math
import torch
from torch.utils.data import Dataset
import os


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
        print("一次性迭代：\n")
        return iter(iterate_data(iter_start, 178945))

# def read_file(path):
#
def iterate_data(head_path, ech_size):
    print("迭代次数")
    count = 10
    iter_start = 0
    char_tst = 0
    iter_end = ech_size
    list = []
    for i in range(178945):
        list.append(i)

    while True:
        result = list[iter_start : iter_start + 128]
        iter_start += 128
        char_tst += 1

        if iter_start >= iter_end:
            break
        yield result, char_tst


# def read_paths(paths):
#     for path in paths:



# def read_txt(label_path, txt_path):



 # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset(start=3, end=79)
now = time.time()
labels = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=128)
print(list(labels))
print("\n", time.time() - now)