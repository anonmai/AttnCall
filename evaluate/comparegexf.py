import datetime
import os
import networkx as nx
from predict import Predict
import xlwings as xl


class Compare:
    def __init__(self, exe, opt_level, ablation_key, split):
        self._exe = exe
        self._opt_level = opt_level
        self._ablation_key = ablation_key
        self._split = split

    def compare(self):
        spec_file = f'../data/evaluate/gexf/{self._opt_level}/{self._exe}.gexf'
        predict_file = f'../data/evaluate/predict/{self._opt_level}/{self._exe}_base.x86_64_linux'
        if self._exe == 'sphinx3':
            predict_file = f'../data/evaluate/predict/{self._opt_level}/sphinx_livepretend_base.x86_64_linux'
        store_file = predict_file.replace('predict', 'graph')
        store_dir = os.path.split(store_file)[0]
        os.makedirs(store_dir, exist_ok=True)
        if not os.path.exists(store_file):
            predict = Predict(ablation_key=self._ablation_key, opt_level=self._opt_level, top_repeat=1, bottom_repeat=1)
            graph = predict.generate_graph(predict_file)
            nx.write_gexf(graph, store_file)
        for split in self._split:
            self._compare(spec_file, store_file, split)

    def _compare(self, pin_file, angr_file, split):
        """
        比较两个gexf文件中的各项数据
        :param pin_file pin产生的gexf文件
        :param angr_file angr产生的gexf文件
        :param split 判断是否合法的分界线
        """

        # G1: pin结果的图
        # G2: angr结果的图
        # G3: angr结果经过过滤后的合法的图
        # G4: G1与G3的共同边组成的图
        # G5: G1经过过滤掉angr结果中不存在的bottom后组成的图
        # G6: G5与G3的共同边组成的图
        # G1，G2由文件加载
        G1 = nx.read_gexf(pin_file)
        G2 = nx.read_gexf(angr_file)
        G3 = nx.DiGraph()
        G4 = nx.DiGraph()
        G5 = nx.DiGraph()
        G6 = nx.DiGraph()

        G1_top_set = set()
        G1_bottom_set = set()
        G2_top_set = set()
        G2_bottom_set = set()
        G3_top_set = set()
        G3_bottom_set = set()
        G4_top_set = set()
        G4_bottom_set = set()
        G5_top_set = set()
        G5_bottom_set = set()
        G6_top_set = set()
        G6_bottom_set = set()

        for u, v in G1.edges:
            G1_top_set.add(u)
            G1_bottom_set.add(v)
            if u in G2.nodes and v in G2.nodes:
                G5_top_set.add(u)
                G5_bottom_set.add(v)
                G5.add_edge(u, v)
        for u, v, wt in G2.edges.data('weight'):
            G2_top_set.add(u)
            G2_bottom_set.add(v)
            if wt > split:
                G3.add_edge(u, v)
                G3_top_set.add(u)
                G3_bottom_set.add(v)
                if G1.has_edge(u, v):
                    G4.add_edge(u, v)
                    G4_top_set.add(u)
                    G4_bottom_set.add(v)
                if G5.has_edge(u, v):
                    G6.add_edge(u, v)
                    G6_top_set.add(u)
                    G6_bottom_set.add(v)

        # G1
        pin_top = len(G1_top_set)
        pin_bottom = len(G1_bottom_set)
        pin_edge = len(G1.edges)

        # G2
        angr_top = len(G3_top_set)
        angr_bottom = len(G3_bottom_set)
        angr_edge = len(G3.edges)

        # G4
        common_top = len(G4_top_set)
        common_bottom = len(G4_bottom_set)
        common_edge = len(G4.edges)

        if pin_edge == 0:
            recall = -1
        else:
            recall = float(common_edge) / float(pin_edge)

        # G5
        f_pin_top = len(G5_top_set)
        f_pin_bottom = len(G5_bottom_set)
        f_pin_edge = len(G5.edges)

        # G6
        f_common_top = len(G6_top_set)
        f_common_bottom = len(G6_bottom_set)
        f_common_edge = len(G6.edges)

        if f_pin_edge == 0:
            f_recall = -1
        else:
            f_recall = float(f_common_edge) / float(f_pin_edge)

        opt_level = os.path.split(os.path.split(pin_file)[0])[1]
        aict = self.compute_aict(G2, split)
        self.add_row_to_xlsx([
            datetime.date.today(),
            os.path.split(pin_file)[1],
            opt_level,
            split,
            pin_top,
            pin_bottom,
            pin_edge,
            angr_top,
            angr_bottom,
            angr_edge,
            common_top,
            common_bottom,
            common_edge,
            recall,
            f_pin_top,
            f_pin_bottom,
            f_pin_edge,
            f_common_top,
            f_common_bottom,
            f_common_edge,
            f_recall,
            aict,
        ])

    def add_row_to_xlsx(self, row: list):
        """
        在excel中增加一行row
        :param row: row是一个列表，这个列表的格式是
        [
            时间，
            程序名,
            优化等级,
            split,
            pin-top数量，
            pin-bottom数量，
            pin-edge数量，
            angr-top数量，
            angr-bottom数量，
            angr-edge数量，
            共同-top数量，
            共同-bottom数量，
            共同edge数量,
            recall,
            f_pin_top,
            f_pin_bottom,
            f_pin_edge,
            f_common_top,
            f_common_bottom,
            f_common_edge,
            f_recall,
            aict,
        ]
        :return:
        """
        # 1.读取excel文件，如果不存在，则创建一个新的
        xlsx_dir = '../data/evaluate/compare'
        os.makedirs(xlsx_dir, exist_ok=True)
        book_path = os.path.join(xlsx_dir, 'compare.xlsx')
        app = xl.App(visible=False, add_book=False)
        book = app.books.add()
        if os.path.exists(book_path):
            book = app.books.open(book_path)

        # 2.excel的sheet
        sheet: xl.Sheet = book.sheets[r'sheet1']
        sheet.range('A1', 'V1').column_width = 18
        if not sheet.range('A1').value:
            # 2.1 如果这个excel第一行不存在，创建第一行
            sheet.range('A1').value = [
                'date',
                'program',
                'opt Level',
                'split',
                'pin-top',
                'pin-bottom',
                'pin-edge',
                'angr-top',
                'angr-bottom',
                'angr-edge',
                'common-top',
                'common-bottom',
                'common-edge',
                'recall',
                'f_pin_top',
                'f_pin_bottom',
                'f_pin_edge',
                'f_common_top',
                'f_common_bottom',
                'f_common_edge',
                'f_recall',
                'aict',
            ]

        # 3.获得要写入内容的行数
        target_row = sheet.used_range.last_cell.row + 1

        # 4.写入row并保存内容
        sheet.range(f'A{target_row}').value = row
        book.save(book_path)
        book.close()
        print('finish exe =', row[1])

    def compute_aict(self, graph: nx.DiGraph, split):
        target_num = 0
        call_site_set = set()
        for u, v, wt in graph.edges.data('weight'):
            if wt > split:
                call_site_set.add(u)
                target_num += 1
        aict = target_num // len(call_site_set)
        return aict


if __name__ == '__main__':
    exes = [
        'perlbench',
        'bzip2',
        'gcc',
        'gobmk',
        'h264ref',
        'hmmer',
        'milc',
        'sjeng',
        'sphinx3',
    ]
    for exe in exes:
        compare = Compare(exe, 'O0', 'ano', [0.1, 0.2, 0.3, 0.4, 0.5])
        compare.compare()
    for exe in exes:
        compare = Compare(exe, 'O3', 'ano', [0.1, 0.2, 0.3, 0.4, 0.5])
        compare.compare()
