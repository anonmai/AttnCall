import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class HighAttentionOp:
    def __init__(self, record_file):
        self._record_file = record_file
        self._top_mne_dic = {}
        self._top_op_dic = {}
        self._bottom_mne_dic = {}
        self._bottom_op_dic = {}
        self._top_mne_count = 0
        self._bottom_mne_count = 0
        self._top_op_count = 0
        self._bottom_op_count = 0
        self._count_dict = dict()

        self._top_op_percent_dict = {}
        self._bottom_op_percent_dict = {}
        self._top_op_percent_count = 0
        self._bottom_op_percent_count = 0

    @property
    def record_file(self):
        return self._record_file

    def count_top_mne(self, record):
        self._top_mne_count += 1
        mne = record[0]
        if record[1] != ' ':
            mne += record[1]
        count = self._top_mne_dic.get(mne)
        if count:
            self._top_mne_dic[mne] = count + 1
        else:
            self._top_mne_dic[mne] = 1

    def count_top_op(self, record):
        self._top_op_count += 1
        op = record[1]
        if record[0] == 'mem':
            op = '(' + op + ')'
        count = self._top_op_dic.get(op)
        if count:
            self._top_op_dic[op] = count + 1
        else:
            self._top_op_dic[op] = 1

    def count_bottom_mne(self, record):
        self._bottom_mne_count += 1
        mne = record[0]
        if record[1] != ' ':
            mne += record[1]
        count = self._bottom_mne_dic.get(mne)
        if count:
            self._bottom_mne_dic[mne] = count + 1
        else:
            self._bottom_mne_dic[mne] = 1

    def count_bottom_op(self, record):
        self._bottom_op_count += 1
        op = record[1]
        if record[0] == 'mem':
            op = '(' + op + ')'
        count = self._bottom_op_dic.get(op)
        if count:
            self._bottom_op_dic[op] = count + 1
        else:
            self._bottom_op_dic[op] = 1

    def count_top_op_percent(self, record):
        self._top_op_percent_count += 1
        op = record[1]
        if record[0] == 'mem':
            op = '(' + op + ')'
        count = self._top_op_percent_dict.get(op)
        if count:
            self._top_op_percent_dict[op] = count + 1
        else:
            self._top_op_percent_dict[op] = 1

    def count_bottom_op_percent(self, record):
        self._bottom_op_percent_count += 1
        op = record[1]
        if record[0] == 'mem':
            op = '(' + op + ')'
        count = self._bottom_op_percent_dict.get(op)
        if count:
            self._bottom_op_percent_dict[op] = count + 1
        else:
            self._bottom_op_percent_dict[op] = 1

    def load_data(self):
        with open(self.record_file, mode='r') as f:
            data = json.load(f)
        return data

    def count_data(self):
        data = self.load_data()
        for records in data:
            if len(records[0]) == 2:
                self.count_top_mne(records[0])

            elif len(records[0]) == 5:
                self.count_top_op(records[0])

            if len(records[1]) == 2:
                self.count_bottom_mne(records[1])
            elif len(records[1]) == 5:
                self.count_bottom_op(records[1])
            if len(records[0]) == 5 and len(records[1]) == 5:
                self.count_top_op_percent(records[0])
                self.count_bottom_op_percent(records[1])

        result_file = f'../data/hightattentionop/{os.path.split(self.record_file)[1]}'
        os.makedirs(os.path.split(result_file)[0], exist_ok=True)
        with open(result_file, mode='w') as f:
            # json.dump([self.top_mne_dic, self.top_op_dic, self.bottom_mne_dic, self.bottom_op_dic], f)
            json.dump({
                'top_mne': self.top_count_of_dict(self._top_mne_dic, 11),
                'top_op': self.top_count_of_dict(self._top_op_dic, 11),
                'bottom_mne': self.top_count_of_dict(self._bottom_mne_dic, 11),
                'bottom_op': self.top_count_of_dict(self._bottom_op_dic, 11)
            }, f)

    def top_count_of_dict(self, dic: dict, count=10):
        top_keys = sorted(dic, key=dic.get, reverse=True)[:count]
        value_sum = sum(dic.values())
        result = []
        for key in top_keys:
            result.append((key, dic.get(key), value_sum))
        return result

    def draw_bar(self):
        file = f'../data/hightattentionop/{os.path.split(self.record_file)[1]}'
        with open(file, mode='r') as f:
            datas = json.load(f)

        # 绘制饼状图
        fig, axs = plt.subplots(2, 2)
        labels = [['top_mne', 'top_op'], ['bottom_mne', 'bottom_op']]
        for i, sub_labels in enumerate(labels):
            for j, label in enumerate(sub_labels):
                data = datas.get(label)
                sizes = []
                keys = []
                c = 0
                for key, num, count in data:
                    keys.append(key)
                    sizes.append(num)
                    c = count
                size_sum = sum(sizes)
                sizes.append(c - size_sum)
                keys.append('others')
                axs[i, j].pie(sizes, labels=keys, autopct='%1.1f%%', startangle=90)
                axs[i, j].axis('equal')
        plt.show()

    def find_max_percent(self):
        data = self.load_data()
        # top_keys = map(lambda x: x[0], self.top_count_of_dict(self._top_op_percent_dict, 11))
        # top_keys = list(filter(lambda x: x != ' ', top_keys))[:10]
        # bottom_keys = map(lambda x: x[0], self.top_count_of_dict(self._bottom_op_percent_dict, 11))
        # bottom_keys = list(filter(lambda x: x != ' ', bottom_keys))[:10]
        top_keys, bottom_keys = self.get_keys()
        count_dic = self._count_dict
        for top in top_keys:
            for bottom in bottom_keys:
                count_dic[(top, bottom)] = 0
        count_dic['pass'] = 0
        for records in data:
            if len(records[0]) != 5 or len(records[1]) != 5:
                continue
            top_key = self.key_op(records[0])
            bottom_key = self.key_op(records[1])
            if top_key not in top_keys or bottom_key not in bottom_keys:
                self.add_to_dict(count_dic, 'pass')
                continue
            else:
                self.add_to_dict(count_dic, (top_key, bottom_key))

    def add_to_dict(self, count_dic, key):
        count = count_dic[key]
        count += 1
        count_dic[key] = count

    def key_op(self, record):
        if record[0] == 'mem' and record[1] != ' ':
            return '(' + record[1] + ')'
        elif record[1] != ' ':
            return record[1]

    def draw_percent(self):
        self.find_max_percent()
        count_dict = self._count_dict
        # 计算总和
        total = sum(count_dict.values())
        del count_dict['pass']

        top_keys, bottom_keys = self.get_keys()

        # 创建一个数据表，其中的元素为C中对应key的value的百分比
        data = pd.DataFrame(index=bottom_keys, columns=top_keys)
        for (i, j), value in count_dict.items():
            data.loc[j, i] = value / total * 100  # 计算百分比
        data = data.astype(float)
        # 显示数据表
        sns.heatmap(data, annot=True, cmap='Blues')
        plt.show()

    def get_keys(self):
        # top_keys = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9', 'edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d']
        # bottom_keys = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9', 'edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d']
        top_keys = map(lambda x: x[0], self.top_count_of_dict(self._top_op_percent_dict, 11))
        top_keys = list(filter(lambda x: x != ' ', top_keys))[:10]
        bottom_keys = map(lambda x: x[0], self.top_count_of_dict(self._bottom_op_percent_dict, 11))
        bottom_keys = list(filter(lambda x: x != ' ', bottom_keys))[:10]
        return top_keys, bottom_keys


if __name__ == '__main__':
    high_att = HighAttentionOp('../data/statistics/ano/O1/record')
    high_att.count_data()
    high_att.draw_bar()
    # high_att.draw_percent()
