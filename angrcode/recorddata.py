import time
import utils.angroperation as angrop
from angrcode import symbolstrategy
import bisect
import os


class RecordData:
    def __init__(self, queue, file, count_list, strategy=None, subdir='train', ablation_key_list=None):
        self._file = file
        self._queue = queue
        self._strategy = strategy
        self._subdir = subdir
        self._count_list = count_list
        if not ablation_key_list:
            ablation_key_list = []
        self._ablation_key_list = ablation_key_list
        if not self._strategy:
            self._strategy = symbolstrategy.default_strategy(file, ablation_key_list)

    @property
    def file(self):
        return self._file

    @property
    def queue(self):
        return self._queue

    @property
    def strategy(self) -> symbolstrategy.SymbolStrategy:
        return self._strategy

    def put_data(self, pre, suc, insns_addr_list, node_map, record_path_list: list[str]):
        """
        记录数据到step2文件夹
        :param pre: top部分的指令数组
        :param suc: bottom部分的指令数组
        :param insns_addr_list: 所有合法指令的列表
        :param node_map: 指令和它所在node的字典
        :param record_path_list: 记录数据的路径，是列表，列表中所有路径都会记录数据
        :return:
        """
        count = 10
        project = self.strategy.context.project
        cfg_graph = self.strategy.context.graph
        functions = self.strategy.context.function_graph
        # 把正例写入文件用于训练
        data = self.generate_step2_data(pre, suc, label='1')

        while self.queue.qsize() > 10000:
            time.sleep(1)

        self.queue.put((record_path_list, data))

        # 把反例写入文件用于训练
        # 随机地址开始
        for i in range(1):
            neg_insns = angrop.random_insns(project, cfg_graph, count, False, insns_addr_list, node_map)
            if len(neg_insns) == 0:
                continue
            data = self.generate_step2_data(pre, neg_insns, label='0')
            self.queue.put((record_path_list, data))
        # 以一个block的开头作为开头
        for i in range(1):
            neg_insns = angrop.random_insns(project, cfg_graph, count, True, insns_addr_list, node_map)
            if len(neg_insns) == 0:
                continue
            data = self.generate_step2_data(pre, neg_insns, label='0')
            self.queue.put((record_path_list, data))
        # 以一个函数的开头作为开头
        for i in range(4):
            neg_insns = angrop.random_insns(project, cfg_graph, count, True, insns_addr_list, node_map, True, functions)
            if len(neg_insns) == 0:
                continue
            data = self.generate_step2_data(pre, neg_insns, label='0')
            self.queue.put((record_path_list, data))

    def step0_to_step2(self):
        """
        一个project从step0到step2的全过程
        :return:
        """
        print('file =', self.file)

        # 计算record_data需要用到的参数
        node_map = self.strategy.context.node_map
        insns_addr_list = self.insns_addr_list()
        optlevel, index = self.get_optlevel()
        optcount = self._count_list[index]
        file_list = self.step2_record_file_path(optcount, optlevel)

        # step1-step2
        for pre, suc in self.strategy.generate_inst():
            # 每record_data一次，就要在对应优化等级的计数器上+1
            self._count_list[index] += 1
            self.put_data(pre, suc, insns_addr_list, node_map, file_list)

    def generate_step2_data(self, top, bottom, label):
        """
        step2记录内容构造
        :param top: top指令数组
        :param bottom: bottom指令数组
        :param label: 标签
        :return:
        """
        op_list_top = []
        op_list_bottom = []
        for asm_insn in top:
            ops = self.strategy.translate_insn(asm_insn)
            op_list_top.extend(ops)
        for asm_insn in bottom:
            ops = self.strategy.translate_insn(asm_insn)
            op_list_bottom.extend(ops)
        data = ' '.join(op_list_top) + '\n' + ' '.join(op_list_bottom) + '\n' + label + '\n'
        return data

    def insns_addr_list(self) -> list[int]:
        """
        获得所有合法指令的列表
        :return:
        """
        min_addr, max_addr = self.strategy.keep_address_range()
        insns_list = []
        for node in self.strategy.context.graph:
            for addr in node.instruction_addrs:
                if min_addr <= addr <= max_addr:
                    bisect.insort(insns_list, addr)
        return insns_list

    def get_optlevel(self) -> (str, int):
        """
        从文件名中获得优化等级
        :return: 优化等级和优化等级所处位置 如 O1的index是1
        """
        file_name = os.path.split(self.file)[1]
        opt_level = ['O0', 'O1', 'O2', 'O3']
        for i, opt in enumerate(opt_level):
            if '-' + opt in file_name:
                return opt, i

    def step2_record_file_path(self, optcount, optlevel) -> list[str]:
        """
        根据计数决定数据存放的文件路径
        :param optcount: 当前优化等级的计数
        :param optlevel: 当前优化等级
        :return:
        """

        path_opt_count = optcount // 20000
        path_opt = os.path.join(self.step2_dir(), optlevel)
        os.makedirs(path_opt, exist_ok=True)
        file_opt = os.path.join(path_opt, 'step2-' + optlevel + str(path_opt_count))

        return [file_opt]

    def step2_dir(self):
        step2 = f'../data/step2/{self.subdir()}/{self.strategy.ablation_factory.ablation_key(self._ablation_key_list)}/'
        if not os.path.exists(step2):
            os.makedirs(step2, exist_ok=True)
        return step2

    def subdir(self) -> str:
        """
        用于区分文件夹中的数据是用于test还是用于train
        :return:
        """
        return self._subdir
