import json
import os
import keras
import utils.fileutil
from transformer.f1score import F1Score
from transformer.tokengeneration import legal_token_list
import tensorflow as tf
from transformer.const import MAX_LENGTH_TOP, MAX_LENGTH_BOTTOM
from tqdm import tqdm


class InsnStatistics:
    def __init__(self, key, opt_level):
        self._model = None
        self._ablation_key = key
        self._opt_level = opt_level

    @property
    def ablation_key(self):
        return self._ablation_key

    @property
    def opt_level(self):
        return self._opt_level

    def model_v2(self):
        if not self._model:
            path = f"../data/model2/train/{self.ablation_key}/{self.opt_level}"
            self._model = keras.models.load_model(path, custom_objects={'F1Score': F1Score})
        return self._model

    def step2_files(self):
        files = utils.fileutil.get_files(f'../data/step2/train/{self.ablation_key}/{self.opt_level}/')
        return files
        # return [f'../data/step2/train/{self.ablation_key}/{self.opt_level}/step2-{self.opt_level}0']

    def load_pairs(self):
        pairs = []
        for file in self.step2_files():
            with open(file, mode='r') as f:
                line1 = f.readline()
                count = 1000
                while line1 or count > 0:
                    count -= 1
                    line2 = f.readline()
                    line3 = f.readline()
                    if line3 == '1\n':
                        top = line1.split()
                        bottom = line2.split()
                        pairs.append((top, bottom))
                    line1 = f.readline()
            print(file)
        return pairs

    def get_layer(self):
        """
        获得embed的layer
        :return:
        """
        tokens = legal_token_list(self.ablation_key, self.opt_level)
        layer = tf.keras.layers.TextVectorization(vocabulary=tokens)
        return layer

    def translate_input(self, pair):
        top, bottom = pair
        layer = self.get_layer()
        top = layer(top)
        top = tf.squeeze(top, axis=-1)
        bottom = layer(bottom)
        bottom = tf.squeeze(bottom, axis=-1)

        length = len(top)
        top = tf.pad(top, tf.constant([[1, 0]]) * (MAX_LENGTH_TOP - length))
        length = len(bottom)
        bottom = tf.pad(bottom, tf.constant([[0, 1]]) * (MAX_LENGTH_BOTTOM - length))

        top = tf.reshape(top, [-1, MAX_LENGTH_TOP])
        bottom = tf.reshape(bottom, [-1, MAX_LENGTH_BOTTOM])
        return top, bottom

    def statistics(self, pair):
        model = self.model_v2()
        top, bottom = self.translate_input(pair)
        input_tensor = tf.concat([top, bottom], axis=1)
        model(input_tensor, training=False)
        left_embed_layer = model.pos_embedding1
        left_layer = model.enc_layers1[-1]
        right_embed_layer = model.pos_embedding2
        right_layer = model.enc_layers2[-1]

        left_embed = left_embed_layer(top, training=False)
        k = left_layer(left_embed, training=False)
        q = right_embed_layer(bottom, training=False)

        _, attention = right_layer.self_attention.mha(query=q, key=k, value=q, return_attention_scores=True)
        attention = tf.squeeze(attention, axis=0)
        top_str, bottom_str = pair
        top_k_token_pairs = self.calculate_attention(attention, top_str, bottom_str)
        return top_k_token_pairs

    def calculate_attention(self, attention, top, bottom):
        top_tokens = self.get_tick_labels(top)
        bottom_tokens = self.get_tick_labels(bottom)

        tensors = tf.split(attention, num_or_size_splits=attention.shape[0], axis=0)

        # 将分割后的张量相加
        sum_tensor = tf.reduce_sum(tensors, axis=0)

        # 移除多余的维度
        sum_tensor = tf.squeeze(sum_tensor)

        # 找到 (30, 30) 张量中最大的 10 个值及其位置
        top_k_values, top_k_indices = tf.math.top_k(tf.reshape(sum_tensor, [-1]), k=10)

        top_k_coordinates = tf.unravel_index(top_k_indices, dims=[30, 30])
        coordinates = tf.transpose(top_k_coordinates).numpy()
        top_k_attention = []
        for i, (raw, col) in enumerate(coordinates):
            top_k_attention.append((top_tokens[col] if len(top_tokens) > col else [' '], bottom_tokens[raw] if len(bottom_tokens) > raw else [' '], str(top_k_values[i].numpy())))
        return top_k_attention

    def get_tick_labels(self, tick: list[str]):
        """
        获得轴上的文字序列的列表
        :param tick: 字符串列表，用来构成轴
        :return:
        """
        tick_labels = []
        for i in range(0, len(tick), 12):
            tick_labels.extend(self.label_row(tick[i: i + 12]))
        return tick_labels

    def label_row(self, words: list[str]) -> list[list]:
        """
        获得图表里的坐标轴上的标记
        :param words: 字符串列表
        :return:
        例如：
        ['mov','pad','pad','rbp','pad','pad','pad,'pad','pad','pad']
        经过转换会变为
        'mov rbp'
        """
        words = list(map(lambda x: ' ' if x == 'pad' else x, words))
        row1 = words[0:2]
        row2 = words[2:7]
        row3 = words[7:]
        rows = [row1, row2, row3]
        return rows

    def record_to_file(self):
        pairs = self.load_pairs()
        record = []
        for pair in tqdm(pairs):
            record.extend(self.statistics(pair))
        record_file = f'../data/statistics/{self.ablation_key}/{self.opt_level}/record'
        os.makedirs(os.path.split(record_file)[0], exist_ok=True)
        with open(record_file, mode='w') as f:
            json.dump(record, f)


if __name__ == '__main__':
    insn_statistics = InsnStatistics('ano', 'O3')
    insn_statistics.record_to_file()
