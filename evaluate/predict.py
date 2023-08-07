import os
import angrcode.angrcontext
import angrcode.symbolstrategy
import transformer.tokengeneration
import angr
import utils.angroperation as arop
from tensorflow import keras
import tensorflow as tf
import networkx as nx
from transformer.const import MAX_LENGTH_TOP, MAX_LENGTH_BOTTOM
import xlwings as xl
from utils import fileutil
from capstone import CsInsn, x86_const
from transformer.f1score import F1Score

predict_file_dir = '../data/evaluate/predict/'
predict_graph_dir = '../data/evaluate/graph/'
ict_dir = '../data/evaluate/ict/'


class Predict:
    def __init__(self, ablation_key, opt_level, top_repeat=1, bottom_repeat=1):
        self._ablation_key = ablation_key
        self._opt_level = opt_level
        self._strategy = None
        self._top_repeat = top_repeat
        self._bottom_repeat = bottom_repeat

    @property
    def strategy(self):
        return self._strategy

    @property
    def ablation_key(self):
        return self._ablation_key

    @property
    def opt_level(self):
        return self._opt_level

    @property
    def top_repeat(self):
        return self._top_repeat

    @property
    def bottom_repeat(self):
        return self._bottom_repeat

    def model_dir(self):
        return f'../data/model2/train/{self.ablation_key}/{self.opt_level}'

    def evaluate_cfgdic_path(self, source_file_path) -> str:
        """
        获得cfgdic的存放/加载路径
        :param source_file_path: 原文件路径
        :return:
        """
        cfgdic_path = source_file_path.replace('predict', 'cfgdic')
        os.makedirs(os.path.split(cfgdic_path)[0], exist_ok=True)
        return cfgdic_path

    def store_cfgdic(self, path) -> dict or None:
        """
        保存cfgdic
        :param path:
        :return: 如果是新构造且dump的cfgdic,会返回cfgdic。其他情况返回None
        """
        # 1.错误检查
        proj = arop.load_project(path)
        if proj is None:
            return
        cfg = arop.analyses_cfg(proj)
        if cfg is None:
            return

        # 2.检查是否已经有分析保存好的文件了
        evaluate_file_path = self.evaluate_cfgdic_path(path)
        if os.path.exists(evaluate_file_path):
            return

        # 3.保存分析好的文件到磁盘
        arop.IO_dump_cfgdic(evaluate_file_path, cfg)
        return cfg

    def generate_graph(self, path) -> nx.DiGraph or None:
        """
        生成一个elf文件的调用图
        :param path: 文件路径
        :return: 结果是一个NetworkX图
        """
        # 加载cfg
        proj = angr.Project(path, auto_load_libs=False)
        cfg = self.store_cfgdic(path)

        if cfg is None:
            evaluate_file_path = self.evaluate_cfgdic_path(path)
            if not os.path.exists(evaluate_file_path):
                return
            cfg = arop.IO_load_cfgdic(evaluate_file_path)
        context = angrcode.angrcontext.AngrContext(proj, cfg, store_dic=False)
        strategy = angrcode.symbolstrategy.SymbolStrategy(context, key_list=[self.ablation_key])
        self._strategy = strategy
        node_map = context.node_map

        # 加载model
        tokens = transformer.tokengeneration.legal_token_list(self.ablation_key, self.opt_level)
        layer = tf.keras.layers.TextVectorization(vocabulary=tokens)
        model_save_dir = self.model_dir()
        model = keras.models.load_model(model_save_dir, custom_objects={'F1Score': F1Score})

        # 获得taken_address
        taken_addr = self.address_taken(cfg.graph, proj)
        pre, suc = self.call_site_and_function_start(layer, taken_addr)

        graph = nx.DiGraph()

        v_list = []
        v_tensor_list = []
        for v in suc.keys():
            v_list.append(v)
            v_tensor = suc.get(v)
            v_tensor_list.extend(v_tensor)
        if len(v_list) == 0:
            return None
        for u in pre.keys():
            if not arop.call_filter(proj, u, node_map):
                continue
            u_tensor = pre.get(u)
            for sub_u in u_tensor:
                y = self.predict(sub_u, v_tensor_list, model)
                for v, w in list(zip(v_list, y)):
                    weight = w[0]
                    if weight > 0.95:
                        # print(sub_u)
                        print('lixiangping')
                    if (u, v) not in graph.edges or graph.get_edge_data(u, v)['weight'] < weight:
                        graph.add_edge(u, v, weight=weight)
        # 把信息记录到excel文件中
        # print('icalls =', i_calls, 'aict =', aict)
        # self.add_row_to_xlsx([program, opt_level, instrs, i_calls, aict])
        return graph

    def predict(self, top: tf.Tensor, bottoms: list[tf.Tensor], model: tf.keras.Model) -> list:
        """
        预测top和bottom是否可以有调用关系
        :param top:
        :param bottoms:
        :param model: 用于预测的模型
        :return: bool值
        """
        x = tf.stack(list(map(lambda bottom: tf.concat([top, bottom], axis=0), bottoms)), axis=0)
        # x = tf.reshape(x, [-1, MAX_LENGTH_TOP + MAX_LENGTH_BOTTOM])
        y = model.predict(x)

        return y

    def translate_insn_to_tensor(self, insns: [], istop: bool, layer: keras.layers.TextVectorization) -> tf.Tensor:
        """
        把指令转化为对应的tensor
        :param insns: 指令列表
        :param istop: true是top，false是bottom
        :param layer: embed的layer
        :return: 一个tensor，最终用来保存到字典中去
        """
        # 把指令列表转化为tensor
        x = tf.concat(list(map(lambda insn: layer.call(tf.constant(self._strategy.translate_insn(insn))), insns)), 0)
        # 做embed
        # x = layer.call(x)
        # 去掉多出来的维度
        x = tf.squeeze(x, axis=-1)
        # 获得tensor长度

        if istop:
            # 如果是top，则在前半段补0
            x = x[-MAX_LENGTH_TOP:]
            length = len(x)
            x = tf.pad(x, tf.constant([[1, 0]]) * (MAX_LENGTH_TOP - length))
        else:
            # 如果是bottom，则在后半段补0
            x = x[:MAX_LENGTH_BOTTOM]
            length = len(x)
            x = tf.pad(x, tf.constant([[0, 1]]) * (MAX_LENGTH_BOTTOM - length))
        return x

    def call_site_and_function_start(self, layer, taken_addr) -> (dict, dict):
        """
        找到所有的调用点和函数起点。分别以字典的方式返回，字典的key是地址，value是指令的列表
        :param layer:
        :param taken_addr:
        :return:
        """
        strategy = self._strategy
        functions = strategy.context.functions
        count = 10
        # 调用点的字典
        pre_dic = {}
        # 函数起点的字典
        suc_dic = {}
        project = strategy.context.project
        functions.project = project
        cfg = strategy.context.cfg

        # 找到所有的call调用点，暴力搜索
        indirect_calls = []
        for function in cfg.functions.values():
            function._project = project
            for block in function.blocks:
                for insn in block.capstone.insns:
                    if 'call' in insn.mnemonic and insn.operands[0].type != x86_const.X86_OP_IMM:
                        indirect_calls.append(insn.address)

        for call_site in indirect_calls:
            # 找到调用点的前驱代码
            pre_node, _ = self._strategy.find_cfg_node(call_site)
            x_list = []
            for i in range(self.top_repeat):
                pre = self._strategy.get_pre_insns(pre_node, count)
                if len(pre) > 0:
                    x = self.translate_insn_to_tensor(insns=pre, istop=True, layer=layer)
                    x_list.append(x)
            pre_dic[call_site] = x_list

        # 又是函数起始位置，又是taken_address的所有地址的list
        fun_start_taken_addr = list(filter(lambda func_addr: func_addr in taken_addr, functions))
        for func in fun_start_taken_addr:
            # 找到所有以函数起始点位置为开始的切片
            suc_node, _ = self._strategy.find_cfg_node(func)
            x_list = []
            for i in range(self.bottom_repeat):
                suc = self._strategy.get_suc_insns(suc_node, count)
                if len(suc) > 0:
                    x = self.translate_insn_to_tensor(insns=suc, istop=False, layer=layer)
                    x_list.append(x)
            suc_dic[func] = x_list

        return pre_dic, suc_dic


    def find_data_section(self, proj):
        """
        找到data段
        @param proj: elf文件由angr加载出来的project
        @return: data段
        """
        for section in proj.loader.main_object.sections:
            if section.name == '.data':
                return section
        return None

    def find_immediate_values_in_data(self, proj) -> []:
        """
        找到data段中的立即数
        @param proj: elf文件由angr加载出来的project
        @return:
        """
        # 找到data段
        data_section = self.find_data_section(proj)
        if data_section is None:
            print("No .data section found")
            return []

        # 找到data段的起始地址和结束地址
        data_start = data_section.vaddr
        data_end = data_start + data_section.memsize

        immediate_values = []
        # 每8个字节作为一个立即数，小端方式
        for addr in range(data_start, data_end, 8):
            value = proj.loader.memory.load(addr, 8)
            imm = int.from_bytes(value, 'little')
            immediate_values.append(imm)

        return immediate_values

    def add_row_to_xlsx(self, row: list):
        """
        在excel中增加一行row
        :param row: row是一个列表，这个列表的格式是[程序名, 优化等级, 指令数, 间接跳转数, 平均间接跳转数]
        :return:
        """
        # 1.读取excel文件，如果不存在，则创建一个新的
        os.makedirs(ict_dir, exist_ok=True)
        book_path = os.path.join(ict_dir, 'ict.xlsx')
        app = xl.App(visible=False, add_book=False)
        book = app.books.add()
        if os.path.exists(book_path):
            book = app.books.open(book_path)

        # 2.excel的sheet
        sheet: xl.Sheet = book.sheets[r'sheet1']
        if not sheet.range('A1').value:
            # 2.1 如果这个excel第一行不存在，创建第一行
            sheet.range('A1').value = ['Program', 'Opt Level', 'Instrs', 'I-Calls', 'AICT']

        # 3.获得要写入内容的行数
        target_row = sheet.used_range.last_cell.row + 1

        # 4.写入row并保存内容
        sheet.range(f'A{target_row}').value = row
        book.save(book_path)
        book.close()

    def address_taken(self, graph, project) -> set:
        """
        获得address_taken的set
        @param graph: cfg的graph
        @param project:
        @return:
        """
        taken_addr = set()
        # 这是.text段的最大地址最小地址
        min_addr, max_addr = arop.text_segment(project)
        for node in graph.nodes:
            bb = project.factory.block(node.addr, node.size)
            instructions = bb.disassembly.insns
            for insn in instructions:
                insn: CsInsn = insn.insn
                # rip相关的takenaddress
                if 'rip' in insn.op_str:
                    value = 0
                    # 获取 RIP 寄存器的值
                    rip_value = insn.address + insn.size
                    value += rip_value
                    value += insn.disp
                    if min_addr <= value <= max_addr:
                        taken_addr.add(value)
                # 直接数
                for op in insn.operands:
                    if op.type == x86_const.X86_OP_IMM:
                        value = op.imm
                        if min_addr <= value <= max_addr:
                            taken_addr.add(value)
        # .data段的直接数
        immediate = set(filter(lambda imm: min_addr <= imm <= max_addr, self.find_immediate_values_in_data(project)))
        taken_addr = taken_addr | immediate
        return taken_addr

    def store_file(self, path, graph):
        store_file = path.replace('predict', 'graph')
        os.makedirs(os.path.split(store_file)[0], exist_ok=True)
        nx.write_gexf(graph, store_file)


if __name__ == '__main__':
    # 把要预测的文件放入这个文件夹内，会在同目录生成的graph文件夹下得到一个有向图，有向图的边有权重，权重为该top到bottom可达性的预测值
    # files = fileutil.get_files(predict_file_dir)
    files = ['../data/evaluate/predict/O0/perlbench_base.x86_64_linux']
    os.makedirs(predict_graph_dir, exist_ok=True)
    for file in files:
        opt_levels = ['O0', 'O1', 'O2', 'O3']
        opt_level = opt_levels[0]
        for opt in opt_levels:
            if opt in file:
                opt_level = opt

        predict = Predict(ablation_key='ano', opt_level=opt_level, top_repeat=3, bottom_repeat=3)
        graph = predict.generate_graph(file)
        if graph is None:
            print("graph is None")
        else:
            predict.store_file(file, graph)
