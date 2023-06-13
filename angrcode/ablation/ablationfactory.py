from angrcode.ablation.ablationstrategy import AblationStrategy


class AblationFactory:
    def __init__(self):
        self._ablation_dic = {}

    def ablation_key(self, key_list: list):
        key_list = list(set(key_list))
        key_list.sort()
        key = ''
        for k in key_list:
            key += k
        return key

    def ablation(self, key_list: list) -> AblationStrategy:
        key = self.ablation_key(key_list)
        ab = self._ablation_dic.get(key)
        if ab is not None:
            return ab

        fun_dic = {}
        if 'r' in key_list:
            fun_dic['ablation_reg'] = ablation_reg
        if 'k' in key_list:
            fun_dic['ablation_imm'] = ablation_imm
        if 'm' in key_list:
            fun_dic['ablation_mne'] = ablation_mne
        if 'e' in key_list:
            fun_dic['ablation_mem'] = ablation_mem
        if 'n' in key_list:
            fun_dic['ablation_num'] = ablation_num
        if 'a' in key_list:
            fun_dic['ablation_addr'] = ablation_addr
        if 'o' in key_list:
            fun_dic['ablation_offset'] = ablation_offset
        if 'b' in key_list:
            fun_dic['ablation_base'] = ablation_base
        if 'i' in key_list:
            fun_dic['ablation_index'] = ablation_index
        if 's' in key_list:
            fun_dic['ablation_scale'] = ablation_scale

        AblationClass = type(
            'AblationClass',
            (AblationStrategy,),
            fun_dic
        )
        ablation = AblationClass()
        self._ablation_dic[key] = ablation
        return ablation


def ablation_imm(self, imm):
    return 'imm'


def ablation_mne(self, mne):
    return 'mne'


def ablation_reg(self, reg):
    return 'reg'


def ablation_mem(self, mem):
    return 'mem'


def ablation_num(self, num):
    return 'num'


def ablation_addr(self, addr):
    return 'addr'


def ablation_offset(self, offset):
    return 'offset'


def ablation_base(self, base_reg):
    return 'reg'


def ablation_index(self, index_reg):
    return 'reg'


def ablation_scale(self, scale):
    return 'scale'
