import angr
import utils.angroperation as angrop
import networkx
import os
import pickle


def default_context(file):
    project = angr.Project(file, auto_load_libs=False)
    step1_file = file.replace('step0', 'step1')
    if os.path.exists(step1_file):
        context = AngrContext(project)
        context.IO_load_cfgdic(step1_file)
    else:
        context = AngrContext(project, generate_cfg=True, store_dic=True)
    return context


class AngrContext:
    def __init__(self, project=None, cfg=None, generate_cfg=False, store_dic=True):
        self._project = project
        self._graph = None
        self._functions = None
        self._function_graph = None
        self._node_map = {}
        self._store_dic = store_dic
        if cfg is not None:
            self.cfg = cfg
        if generate_cfg and cfg is None:
            self.cfg = angrop.analyses_cfg(self.project)


    @property
    def node_map(self):
        return self._node_map

    @property
    def store_dic(self):
        return self._store_dic

    @store_dic.setter
    def store_dic(self, store_dic: bool):
        self._store_dic = store_dic

    @property
    def project(self):
        return self._project

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg
        cfg.project = self.project
        self.graph = cfg.graph
        self._functions = cfg.functions
        self.function_graph = self.generate_function_graph()
        if self._store_dic:
            self.store_cfgdic(cover=True)

    @property
    def graph(self):
        if self._graph:
            return self._graph
        if self.cfg:
            return self.cfg.graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        if self._graph:
            self.create_node_map()

    @property
    def functions(self):
        if self._functions:
            return self._functions
        if self.cfg:
            return self.cfg.functions

    @property
    def function_graph(self):
        return self._function_graph

    @function_graph.setter
    def function_graph(self, function_graph):
        self._function_graph = function_graph

    def create_node_map(self):
        for node in self.graph.nodes:
            for addr in node.instruction_addrs:
                self.node_map[addr] = node

    def step1_path(self):
        if 'step0' in self.project.filename:
            return self.project.filename.replace('step0', 'step1')
        else:
            raise Exception('file path wrong, step0 not in path')

    def construct_cfgdic(self) -> dict:
        """
        构造 dump cfgdic到磁盘的信息
        :return:
        """
        dic = {
            'graph': self.graph,
            'function_graph': self.function_graph
        }
        return dic

    def generate_function_graph(self) -> networkx.DiGraph:
        function_graph = networkx.DiGraph()
        functions = self.functions
        for func in functions:
            # 找到每个函数对象，函数对象里的调用点既是要找的内容
            function = functions.get_by_addr(func)
            function._project = self.project

            # 把所有调用点处指令片段找出来
            call_sites = list(function.get_call_sites())
            for call_site in call_sites:
                # 偶尔会出现call_site是None的情况，需要排除
                if call_site is None:
                    continue
                call_target = function.get_call_target(call_site)
                function_graph.add_edge(call_site, call_target)
        return function_graph

    def store_cfgdic(self, cover=False) -> dict or None:
        """
        保存cfgdic
        :return: 如果是新构造且dump的cfgdic,会返回cfgdic。其他情况返回None
        """
        step1_file_path = self.step1_path()
        if os.path.exists(step1_file_path) and not cover:
            return

        cfgdic = self.construct_cfgdic()
        self.IO_dump_cfgdic(step1_file_path, cfgdic)
        return cfgdic

    def IO_load_cfgdic(self, file):
        """
        加载已经存好的cfg字典
        :param file: 要保存的文件路径
        :return:
        """
        with open(file, mode='rb') as f:
            cfg_dic = pickle.load(f)
            self.graph = cfg_dic['graph']
            self.function_graph = cfg_dic['function_graph']

    def IO_dump_cfgdic(self, file, cfgdic) -> None:
        """
        dump cfgdic到磁盘
        :param file: 磁盘路径
        :return:
        """
        file_dir = os.path.split(file)[0]
        os.makedirs(file_dir, exist_ok=True)
        with open(file, mode='wb') as f:
            pickle.dump(cfgdic, f)
