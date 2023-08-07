import angr.block as bk
from angrcode import angrcontext
from angrcode.angrcontext import AngrContext
from capstone import CsInsn, x86_const
import networkx
from angr.knowledge_plugins.cfg.cfg_node import CFGNode
import random
from angrcode.ablation.ablationfactory import AblationFactory


def default_strategy(file, key_list):
    context = angrcontext.default_context(file)
    strategy = SymbolStrategy(context, key_list)
    return strategy


class SymbolStrategy:
    def __init__(self, context: AngrContext, key_list=None):
        self.out_code: list = ['ret']
        self._context = context
        self._min_address = None
        self._max_address = None
        self._ablation_factory = AblationFactory()
        self._ablation = self._ablation_factory.ablation(key_list=key_list)

    @property
    def ablation_factory(self):
        return self._ablation_factory

    @property
    def context(self):
        return self._context

    @staticmethod
    def keep_imm_range():
        return -128, 128

    def keep_address_range(self):
        if not self._min_address:
            sections = self.context.project.loader.main_object.sections.raw_list
            for sec in sections:
                if sec.name == '.text':
                    self._min_address, self._max_address = sec.min_addr, sec.max_addr
                    return sec.min_addr, sec.max_addr
        else:
            return self._min_address, self._max_address

    def check_x86_op(self, insn: CsInsn):
        mne: str = insn.mnemonic
        mne_list = mne.split()
        token = ['pad', 'pad']
        for i, m in enumerate(mne_list):
            if i >= 2:
                break
            token[i] = self._ablation.ablation_mne(m)
        for i, op in enumerate(insn.operands):
            if i >= 2:
                break
            # type == 1,寄存器
            if op.type == x86_const.X86_OP_REG:
                token_list = ['pad', self.x86_reg_op(op, insn), 'pad', 'pad', 'pad']
            # type == 2,立即数
            elif op.type == x86_const.X86_OP_IMM:
                token_list = ['pad', 'pad', 'pad', 'pad', self.x86_imm_op(op, insn)]
            # type == 3,间接取址
            elif op.type == x86_const.X86_OP_MEM:
                token_list = self.x86_mem_op(op, insn)
            else:
                raise Exception('wrong x86 insn operand, op.type =', op.type)
            token.extend(token_list)
        token.extend(['pad', 'pad', 'pad', 'pad', 'pad'] * (2 - len(insn.operands)))
        if 'imul' in token[0] and len(insn.operands) >= 3:
            token[5] = 'imul-x'
        return token

    def translate_insn(self, insn: bk.CapstoneInsn) -> list[str]:
        """
        把指令对象处理转化为可使用的字符串
        :param insn: 指令对象
        :return: 输入到文件的字符串
        """
        return self.check_x86_op(insn.insn)

    def x86_mem_op(self, operand, insn):
        result = [self._ablation.ablation_mem('mem'), 'pad', 'pad', 'pad', 'pad']
        if operand.value.mem.base != 0:
            reg = insn.reg_name(operand.value.mem.base)
            result[1] = self._ablation.ablation_base(reg)
        if operand.value.mem.index != 0:
            reg = insn.reg_name(operand.value.mem.index)
            result[2] = self._ablation.ablation_index(reg)
        if operand.value.mem.scale != 0:
            result[3] = self._ablation.ablation_scale(operand.value.mem.scale)
        if operand.value.mem.disp != 0:
            min_imm, max_imm = self.keep_imm_range()
            offset = operand.value.mem.disp
            # 这个if-else的逻辑是：如果offset是小数字，而且保留了小数字，则offset也按照小数字的方式保存
            if min_imm <= offset <= max_imm and self._ablation.ablation_imm(offset) != 'imm':
                result[4] = self._ablation.ablation_imm(offset)
            else:
                result[4] = self._ablation.ablation_offset(operand.value.mem.disp)
        return result

    def x86_imm_op(self, operand, insn):
        imm = operand.imm
        min_imm, max_imm = self.keep_imm_range()
        if min_imm <= imm <= max_imm:
            return self._ablation.ablation_imm(imm)

        min_addr, max_addr = self.keep_address_range()
        if min_addr <= imm <= max_addr:
            return self._ablation.ablation_addr(imm)

        return self._ablation.ablation_num(imm)

    def x86_reg_op(self, operand, insn):
        return self._ablation.ablation_reg(insn.reg_name(operand.value.reg))

    def generate_inst(self) -> (list, list):
        """
        生成pre和suc的指令列表
        :return:
        """
        # 获得所有cfg中指令和node的对应字典
        functions: networkx.DiGraph = self.context.function_graph

        # top和bottom的指令取数
        count = 10
        for call_site in functions.nodes:
            for call_target in functions.successors(call_site):
                # 调用点的前驱代码
                pre_node, _ = self.find_cfg_node(call_site)
                pre = self.get_pre_insns(pre_node, count)

                # 调用点的后继代码
                suc_node, _ = self.find_cfg_node(call_target)
                suc = self.get_suc_insns(suc_node, count)

                if pre_node is None or suc_node is None:
                    continue
                if len(pre) == 0 or len(suc) == 0:
                    continue
                pre = pre[:10]
                suc = suc[:10]
                (yield pre, suc)

    def find_cfg_node(self, addr: int) -> (CFGNode, int):
        node = self.context.node_map.get(addr)
        if node is not None:
            return node, node.instruction_addrs.index(addr)
        return None, 0

    def get_pre_insns(self, cfg_node: CFGNode, count) -> list:
        """
        获得调用点前的汇编代码
        一个block会有多个前驱，只选择了第一个前驱作为前驱
        :param cfg_node: 调用点所在的node
        :param count: 需要向前获得汇编代码的行数
        :return: 汇编代码列表
        """
        if cfg_node is None:
            return []
        bb = self.context.project.factory.block(cfg_node.addr, cfg_node.size)
        instructions = bb.disassembly.insns
        # 过滤掉nop操作的指令
        instructions = list(filter(lambda x: x.mnemonic != 'nop', instructions))

        size = len(instructions)
        if size >= count:
            return instructions[size - count:]

        predecessors = list(self.context.graph.predecessors(cfg_node))
        if len(predecessors) == 0:
            return instructions

        pre_instructions = self.get_pre_insns(random.choice(predecessors), count - len(instructions))
        if len(pre_instructions) == 0 or self.out_code.count(pre_instructions[-1].mnemonic) > 0:
            pre_instructions = []
        pre_instructions.extend(instructions)
        size = len(pre_instructions)
        if size >= count:
            return pre_instructions[size - count:]
        return pre_instructions

    def get_suc_insns(self, cfg_node: CFGNode, count) -> list:
        """
        获得调用点后的汇编代码
        一个block会有多个后继，只选择了第一个后继作为后继
        :param cfg_node: 当前代码所在的node
        :param count: 需要向后获得汇编代码的行数
        :return:
        """
        if cfg_node is None or cfg_node.size == 0:
            return []
        bb = self.context.project.factory.block(cfg_node.addr, cfg_node.size)
        instructions = bb.disassembly.insns
        if len(instructions) == 0:
            return []
        if self.out_code.count(instructions[-1].mnemonic) > 0:
            if len(instructions) > count:
                return instructions[:count]
            return instructions

        successors = list(self.context.graph.successors(cfg_node))
        if len(successors) == 0:
            if len(instructions) > count:
                return instructions[:count]
            return instructions

        size = len(instructions)
        if size >= count:
            return instructions[:count]

        suc_instructions = self.get_suc_insns(random.choice(successors), count - size)
        instructions.extend(suc_instructions)
        size = len(instructions)
        if size >= count:
            return instructions[:count]
        return instructions
