class AblationStrategy:
    def ablation_imm(self, imm):
        return str(imm)

    def ablation_mne(self, mne):
        return mne

    def ablation_reg(self, reg):
        return reg

    def ablation_mem(self, mem):
        return mem

    def ablation_num(self, num):
        return str(num)

    def ablation_addr(self, addr):
        return str(addr)

    def ablation_offset(self, offset):
        return str(offset)

    def ablation_base(self, base_reg):
        return base_reg

    def ablation_index(self, index_reg):
        return index_reg

    def ablation_scale(self, scale):
        return str(scale)
