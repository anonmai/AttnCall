import random
import networkx
from capstone import CsInsn, x86_const
import pickle
import angr


# 函数调用点出现的操作符
out_code: list = ['ret']
# 遗漏的寄存器或者操作码
loss_token = set()

# 唯一定义的词汇表
token_dir = '../data/tokenset'
token_path = '../data/tokenset/token'


def create_node_map(graph) -> {}:
    node_map = {}
    for node in graph.nodes:
        for addr in node.instruction_addrs:
            node_map[addr] = node
    return node_map


def get_pre_insns(proj, cfg_node, count, graph) -> list:
    """
    获得调用点前的汇编代码
    一个block会有多个前驱，只选择了第一个前驱作为前驱
    :param proj: proj代表angr反汇编得到的project
    :param cfg_node: 调用点所在的node
    :param count: 需要向前获得汇编代码的行数
    :param graph: 调用图
    :return: 汇编代码列表
    """
    if cfg_node is None:
        return []
    bb = proj.factory.block(cfg_node.addr, cfg_node.size)
    instructions = bb.disassembly.insns
    # 过滤掉nop操作的指令
    instructions = list(filter(lambda x: x.mnemonic != 'nop', instructions))

    size = len(instructions)
    if size >= count:
        return instructions[size - count:]

    predecessors = list(graph.predecessors(cfg_node))
    if len(predecessors) == 0:
        return instructions

    pre_instructions = get_pre_insns(proj, random.choice(predecessors), count - len(instructions), graph)
    if len(pre_instructions) == 0 or out_code.count(pre_instructions[-1].mnemonic) > 0:
        pre_instructions = []
    pre_instructions.extend(instructions)
    size = len(pre_instructions)
    if size >= count:
        return pre_instructions[size - count:]
    return pre_instructions


def get_suc_insns(project, cfg_node, count, graph) -> list:
    """
    获得调用点后的汇编代码
    一个block会有多个后继，只选择了第一个后继作为后继
    :param project: proj代表angr反汇编得到的project
    :param cfg_node: 当前代码所在的node
    :param count: 需要向后获得汇编代码的行数
    :param graph: cfg调用图
    :return:
    """
    if cfg_node is None or cfg_node.size == 0:
        return []
    bb = project.factory.block(cfg_node.addr, cfg_node.size)
    instructions = bb.disassembly.insns
    if len(instructions) == 0:
        return []
    if out_code.count(instructions[-1].mnemonic) > 0:
        if len(instructions) > count:
            return instructions[:count]
        return instructions

    successors = list(graph.successors(cfg_node))
    if len(successors) == 0:
        if len(instructions) > count:
            return instructions[:count]
        return instructions

    size = len(instructions)
    if size >= count:
        return instructions[:count]

    suc_instructions = get_suc_insns(project, random.choice(successors), count - size, graph)
    instructions.extend(suc_instructions)
    size = len(instructions)
    if size >= count:
        return instructions[:count]
    return instructions


def find_cfg_node(addr: int, node_map: {}) -> (object, int):
    node = node_map.get(addr)
    if node is not None:
        return node, node.instruction_addrs.index(addr)
    return None, 0


def get_call_site_graph(cfg, proj) -> networkx.DiGraph:
    functions = cfg.functions
    function_graph = networkx.DiGraph()
    for func in functions:
        # 找到每个函数对象，函数对象里的调用点既是要找的内容
        function = functions.get_by_addr(func)
        function._project = proj

        # 把所有调用点处指令片段找出来
        call_sites = list(function.get_call_sites())
        for call_site in call_sites:
            # 偶尔会出现call_site是None的情况，需要排除
            if call_site is None:
                continue
            call_target = function.get_call_target(call_site)
            function_graph.add_edge(call_site, call_target)
    return function_graph


def shuffle_data(path, line):
    """
    把path的文件打乱，结果放进new_path
    :param path: 源文件路径
    :param line: 打乱的基础单位
    :return:
    """
    with open(path, mode='r') as f:
        # 把所有内容读入文件
        lines = f.readlines()
        # 获得样本总数
        data_num = len(lines) // line
        # [0, 1, 2, 3, ... , data_num-1]
        x = range(data_num)
        # 把x随机重排
        random_list = random.sample(x, k=len(x))
        # 按照x重新排序后的列表，把原内容写入新文件
    with open(path, mode='w') as f:
        for i in random_list:
            # 每line行作为一个数据
            for j in range(line):
                f.write(lines[i * line + j])


def call_filter(proj, call_site, node_map):
    """
    筛选call
    :param proj:
    :param call_site:
    :param node_map:
    :return:
    """
    return is_call(proj, call_site, node_map) and is_call_undirect(proj, call_site, node_map)


def is_call(proj, call_site, node_map: {}) -> bool:
    """
    检查调用点是否是函数调用点
    :param proj:
    :param call_site: 调用点的地址
    :param node_map: 地址对应node的字典
    :return:
    """
    node, index = find_cfg_node(call_site, node_map)
    if node is None:
        return False
    bb = proj.factory.block(node.addr, node.size)
    insn = bb.disassembly.insns[-1]
    if insn is None:
        return False
    cs_insn: CsInsn = insn.insn
    if 'call' in cs_insn.mnemonic:
        return True
    return False


def is_call_undirect(proj, call_site, node_map: {}) -> bool:
    """
    检查调用点是否是间接调用点
    :param proj:
    :param call_site: 调用点的地址
    :param node_map: 地址对应node的字典
    :return:
    """
    if not is_call(proj, call_site, node_map):
        raise Exception('not a call site')

    node, index = find_cfg_node(call_site, node_map)
    if node is None:
        return False
    bb = proj.factory.block(node.addr, node.size)
    insn = bb.disassembly.insns[-1]
    cs_insn: CsInsn = insn.insn
    # 只要不是立即数，就是间接调用
    if cs_insn.operands[0].type != x86_const.X86_OP_IMM:
        return True
    return False


def bottom_addr_list(proj, functions) -> list:
    """
    获得所有可以作为被调用点的地址
    目前实现是返回所有函数起点的地址
    :param proj:
    :param functions:
    :return:
    """
    min_addr, max_addr = text_segment(proj)
    addr_list = list(filter(lambda func_addr: min_addr <= func_addr <= max_addr, functions))
    return addr_list


def filter_file(key: str, files: []):
    """
    查找所有包含关键字的文件
    :param key: 关键词
    :param files: 文件列表
    :return:
    """
    return list(filter(lambda file: key in file, files))


def IO_load_cfgdic(file) -> dict:
    """
    加载已经存好的cfg字典
    :param file: 要保存的文件路径
    :return:
    """
    with open(file, mode='rb') as f:
        cfg_dic = pickle.load(f)
        return cfg_dic


def IO_dump_cfgdic(file, cfgdic) -> None:
    """
    dump cfgdic到磁盘
    :param file: 磁盘路径
    :return:
    """
    with open(file, mode='wb') as f:
        pickle.dump(cfgdic, f)


def load_project(path) -> angr.Project or None:
    """
    load project,可能会失败，失败会返回None
    :param path: 文件路径
    :return:
    """
    try:
        proj = angr.Project(path, auto_load_libs=False)
        return proj
    except Exception as e:
        print('load_project wrong', 'path =', path)
        return None


def analyses_cfg(proj: angr.Project):
    """
    分析cfg，如果出错，会返回None
    :param proj:
    :return:
    """
    try:
        cfg = proj.analyses.CFGFast()
        return cfg
    except Exception as e:
        print('analyses_cfg wrong', 'project =', proj.filename)
        return None


def random_insns(project: angr.Project, cfg_graph, count: int, block_start: bool, insns_addr_list: list[int], node_map,
                 func_start=False, functions=None) -> list:
    """
    随机取count条连续执行的指令
    :param project: 获得指令的project
    :param cfg_graph: project的cfg
    :param count: 需要获取的指令数目
    :param block_start: 表示取得的指令，是否一定要以block的开头作为开头
    :param func_start: 是否一定从函数起点开始
    :param functions: func_start为true时要用的参数，函数地址数组
    :return: 列表形式的指令数
    """
    # 随机获得一个地址，以这个地址作为起点获得指令
    if func_start:
        insns_addr_list = bottom_addr_list(project, functions)
    random_addr = random.choice(insns_addr_list)
    cfg_node, index = find_cfg_node(random_addr, node_map)

    # 以这种方式获得的指令，都是从block的开始处开始执行的
    if block_start or func_start:
        return get_suc_insns(project, cfg_node, count, cfg_graph)
    else:
        insns = get_suc_insns(project, cfg_node, count + index, cfg_graph)
        if len(insns) > count:
            if len(insns) > index:
                insns = insns[index:]
            else:
                insns = insns[:count]
        return insns


def text_segment(proj) -> (int, int):
    """
    找到proj的.text段的地址
    return .text段的min_addr和max_addr
    """
    sections = proj.loader.main_object.sections.raw_list
    for sec in sections:
        if sec.name == '.text':
            return sec.min_addr, sec.max_addr
    raise Exception('do not have .text, proj.name =', proj.filename)
