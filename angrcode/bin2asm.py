import os.path
import multiprocessing
from queue import Empty
import random

import utils.angroperation as arop
from utils import fileutil
from angrcode.recorddata import RecordData


class Bin2Asm:
    def __init__(self, subdir, ablation_expr):
        self._subdir = subdir
        self._ablation_expr = ablation_expr

    @property
    def subdir(self) -> str:
        """
        用于区分文件夹中的数据是用于test还是用于train
        :return:
        """
        return self._subdir

    @property
    def ablation_expr(self):
        return self._ablation_expr

    def step2_dir(self):
        step2 = f'../data/step2/{self.subdir}'
        if not os.path.exists(step2):
            os.makedirs(step2, exist_ok=True)
        return step2

    def IO_write_step2_data(self, queue: multiprocessing.Queue):
        while True:
            try:
                file_path_list, data = queue.get(block=True, timeout=60)
                for file in file_path_list:
                    with open(file, mode='a+') as f:
                        f.write(data)
            except Empty as e:
                print('except e =', e)
                break

    def step0_to_step2(self, step0_file_path: str, count_list: list[int], queue: multiprocessing.Queue):
        """
        一个project从step0到step2的全过程
        :param queue: 多进程队列
        :param step0_file_path: step0的文件路径
        :param count_list:
        :return:
        """
        for ab in self.ablation_expr:
            record = RecordData(queue, step0_file_path, count_list, ablation_key_list=ab, subdir=self.subdir)
            record.step0_to_step2()

    def async_make_step2_data(self):
        """
        多进程 生成step2的数据
        :return:
        """
        files = fileutil.get_files('../data/step0/')
        random.shuffle(files)
        pool = multiprocessing.Pool(7, maxtasksperchild=5)
        queue = multiprocessing.Manager().Queue()
        count_list = multiprocessing.Manager().list([0, 0, 0, 0, 0])
        for file in files:
            pool.apply_async(func=self.step0_to_step2, args=(file, count_list, queue))
        pool.close()
        record_process = multiprocessing.Process(target=self.IO_write_step2_data, args=(queue,))
        record_process.start()
        pool.join()
        record_process.join()
        print(count_list)

    def shuffle_step2_data(self):
        """
        打乱数据的顺序
        :return:
        """
        files = fileutil.get_files(self.step2_dir())
        for file in files:
            arop.shuffle_data(file, 3)


if __name__ == '__main__':
    # 对应的转换规则
    # 'm' -> mnemonic
    # 'r' -> reg
    # 'n' -> num
    # 'a' -> addr
    # 'k' -> imm 控制num、offset中的小数字
    # 'e' -> mem
    # 'b' -> mem base reg
    # 'i' -> mem index reg
    # 's' -> mem scale
    # 'o' -> men offset
    # 顺序无所谓
    ablation_expr_list = [
        ['a', 'n', 'o'],                     # 缺省消融掉addr，num，offset
        #['a', 'n', 'o', 'm'],               # 消融掉所有操作符
        #['a', 'n', 'o', 'r', 'b', 'i'],     # 消融掉所有寄存器符号
        #['a', 'n', 'o', 'k', 's'],          # 消融掉所有数字
    ]
    bin2asm = Bin2Asm('train_O3_114', ablation_expr_list)
    bin2asm.async_make_step2_data()
    bin2asm.shuffle_step2_data()
