import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from transformer.modelv2 import ModelV2
from transformer.const import MAX_LENGTH_TOP, MAX_LENGTH_BOTTOM
from transformer.tokengeneration import legal_token_list
from transformer.f1score import F1Score
from sklearn.manifold import TSNE
from data.demo import top_str, bottom_str, get_str

class DrawAttention:
    def __init__(self, ablation_key, opt_level):
        self._ablation_key = ablation_key
        self._opt_level = opt_level

    @property
    def ablation_key(self):
        return self._ablation_key

    @property
    def opt_level(self):
        return self._opt_level

    def get_top(self) -> list[str]:
        # top_sequence = 'push pad pad rbp pad pad pad pad pad pad pad pad mov pad pad rbp pad pad pad pad rsp pad pad pad sub pad pad rsp pad pad pad pad pad pad pad 16 lea pad pad rax pad pad pad mem rip pad 1 offset mov pad mem rbp pad 1 -8 pad rax pad pad pad mov pad pad rax pad pad pad pad rbp pad 1 -8 mov pad pad esi pad pad pad pad pad pad pad 3 mov pad pad edi pad pad pad pad pad pad pad 2 call pad pad rax pad pad pad pad pad pad pad pad'

        # top_sequence, _ = get_str()
        top_sequence = top_str()
        # top_sequence = 'test pad pad eax pad pad pad pad eax pad pad pad setne pad pad al pad pad pad pad pad pad pad pad movzx pad pad eax pad pad pad pad al pad pad pad test pad pad rax pad pad pad pad rax pad pad pad je pad pad pad pad pad addr pad pad pad pad pad cmp pad mem rbp pad 1 offset pad pad pad pad 0 je pad pad pad pad pad addr pad pad pad pad pad mov pad pad rax pad pad pad mem rbp pad 1 offset mov pad pad rdi pad pad pad pad rax pad pad pad call pad pad pad pad pad addr pad pad pad pad pad'
        top = top_sequence.split()
        num = len(top)
        pad = [''] * (120 - num)
        top = pad + top
        return top
        # return ['pad'] * 120

    def get_bottom(self) -> list[str]:
        # bottom_sequence = 'endbr64 pad pad pad pad pad pad pad pad pad pad pad mov pad pad rbp pad pad pad pad rsp pad pad pad sub pad pad rsp pad pad pad pad pad pad pad 16 mov pad mem rbp pad 1 -4 pad edi pad pad pad mov pad mem rbp pad 1 -8 pad esi pad pad pad mov pad pad edx pad pad pad mem rbp pad 1 -4 mov pad pad eax pad pad pad mem rbp pad 1 -8 add pad pad eax pad pad pad pad edx pad pad pad mov pad pad esi pad pad pad pad eax pad pad pad'
        # _, bottom_sequence = get_str()
        bottom_sequence = bottom_str()
        # bottom_sequence = 'push pad pad rbp pad pad pad pad pad pad pad pad mov pad pad rbp pad pad pad pad rsp pad pad pad sub pad pad rsp pad pad pad pad pad pad pad 16 mov pad mem rbp pad 1 offset pad rdi pad pad pad mov pad pad rax pad pad pad mem rbp pad 1 offset mov pad pad rdx pad pad pad mem rip pad 1 offset mov pad pad rsi pad pad pad pad rdx pad pad pad mov pad pad rdi pad pad pad pad rax pad pad pad call pad pad pad pad pad addr pad pad pad pad pad push pad pad rbp pad pad pad pad pad pad pad pad'
        bottom = bottom_sequence.split()
        return bottom
        # return ['pad'] * 120

    def get_first_top_attention(self):
        model = self.get_modelv2()

        # 假设 text_vectorizer 是你的 TextVectorization 对象
        text_vectorizer = self.get_layer()
        # text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
        # 假设我们有一个文本列表
        with open('../data/tokenset/ano/token-2') as f:
            texts = json.load(f)

        # 将文本转换为整数索引
        vocab_indices = text_vectorizer(texts)

        # 假设 embedding_layer 是你的 Embedding 层
        embedding_layer = model.pos_embedding1.word_embedding

        # 通过 Embedding 层，将词汇表索引转换为向量
        embeddings = embedding_layer(vocab_indices)
        embeddings = tf.squeeze(embeddings, axis=1)

        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, random_state=10, perplexity=6)
        embeddings_2d = tsne.fit_transform(embeddings)

        # 绘制 t-SNE 结果
        plt.figure(figsize=(10, 10))
        for i, token in enumerate(texts):
            c = self.color_label(token)
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=c)
            plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        plt.show()

    def color_label(self, word):
        red_word = {'edx', 'esi', 'ecx', 'rcx', 'r9', 'r9d', 'xor', 'r8d', 'r8', 'rsi', 'rdx', 'num'}
        blue_word = {'jmp', 'ja', 'jb','jae','test','cmp','jl','je','jg','add','jns','jbe','jne','js'}
        green_word = {'al', 'r12d', 'rbx','ebx', 'bl', 'ebp', 'r11d', 'r14d', 'ah', 'r14b', 'r9b', 'jp', 'r12b','r13b','r15b','cx','di','dx','dl','rbp','rax','r8b','r12','bx','r10b','bp','r12w','r13w','r8w','bh','r9w','r10w','rsp','r15w','r11w'}
        yello_word = {'mem','endbr64','sub','and','call','pop','or','men','addr','xorpd','xorps', 'setge','nop','loop','subps','hlt','leave','cli','syscall','sets','shlx','pause','ret','in','int3','shrd','shrx','enter','out'}
        if word in red_word:
            return 'r'
        elif word in blue_word:
            return 'b'
        elif word in green_word:
            return 'g'
        elif word in yello_word:
            return 'y'
        else:
            return 'r'

    def get_label(self):
        return 1

    def get_layer(self):
        """
        获得embed的layer
        :return:
        """
        tokens = legal_token_list(self.ablation_key)
        layer = tf.keras.layers.TextVectorization(vocabulary=tokens)
        return layer

    def get_ticklabels(self, tick: list[str]):
        """
        获得轴上的文字序列的列表
        :param tick: 字符串列表，用来构成轴
        :return:
        """
        ticklabels = []
        for i in range(0, len(tick), 12):
            ticklabels.extend(self.plot_tick_label_row(tick[i: i + 12]))
        return ticklabels

    def plot_tick_label_row(self, words: list[str]) -> list[str]:
        """
        获得图表里的坐标轴上的标记
        :param words: 字符串列表
        :return:
        例如：
        ['mov','pad','pad','rbp','pad','pad','pad,'pad','pad','pad']
        经过转换会变为
        'mov rbp'
        """
        row1 = words[0:2]
        row2 = words[2:7]
        row3 = words[7:]
        rows = [row1, row2, row3]
        for i, row in enumerate(rows):
            rows[i] = list(filter(lambda x: x != 'pad', row))
            if len(rows[i]) == 0:
                rows[i] = ' '
            rows[i] = ' '.join(rows[i])
        return rows


    def get_input_v2(self):
        top = self.get_top()
        bottom = self.get_bottom()
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
        bottom = tf.reshape(bottom, [-1,  MAX_LENGTH_BOTTOM])
        return top, bottom

    def get_modelv2(self) -> ModelV2:
        path = f"../data/model2/train/{self.ablation_key}/{self.opt_level}"
        model = keras.models.load_model(path, custom_objects={'F1Score': F1Score})
        return model

    def modelv2_plot(self):
        top, bottom = self.get_input_v2()
        model = self.get_modelv2()
        model(tf.concat([top, bottom], axis=1), training=False)
        model.summary()
        left_embed_layer = model.pos_embedding1
        left_layer = model.enc_layers1[-1]
        right_embed_layer = model.pos_embedding2
        right_layer = model.enc_layers2[-1].self_attention

        left_embed = left_embed_layer(top, training=False)
        k = left_layer(left_embed, training=False)
        right_embed = right_embed_layer(bottom, training=False)
        q = right_layer(right_embed, training=False)

        _, attention = model.enc_layers2[-1].cross_attention.mha(query=q, key=k, value=q, return_attention_scores=True)
        attention = tf.squeeze(attention, axis=0)
        self.plot_attention_head_v2(attention)

    def plot_attention_head_v2(self, attention):
        top_tokens = self.get_ticklabels(self.get_top())
        bottom_tokens = self.get_ticklabels(self.get_bottom())
        fig = plt.figure(figsize=(40, 10))

        tensors = tf.split(attention, num_or_size_splits=attention.shape[0], axis=0)

        # 将分割后的张量相加
        sum_tensor = tf.reduce_sum(tensors, axis=0)
        sum_tensor = tf.squeeze(sum_tensor, axis=0)

        resized_tensor = tf.reduce_mean(tf.reshape(sum_tensor, (10, 3, 10, 3)), axis=[1, 3])
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.matshow(resized_tensor, cmap='Blues')
        cbar = fig.colorbar(cax)

        for i in range(resized_tensor.shape[0]):
            for j in range(resized_tensor.shape[1]):
                ax.text(j, i, f'{resized_tensor[i, j]:.2f}', va='center', ha='center', color='black')

        top_tokens = self.resize_token(top_tokens)
        bottom_tokens = self.resize_token(bottom_tokens)
        print(resized_tensor)

        # ax.matshow(sum_tensor)
        ax.set_xticks(range(len(top_tokens)))
        ax.set_xticklabels(top_tokens, rotation=90)
        ax.set_yticks(range(len(bottom_tokens)))
        ax.set_yticklabels(bottom_tokens)
        plt.tight_layout(pad=10)

        # for h, head in enumerate(attention):
        #     ax = fig.add_subplot(1, 4, h + 1)
        #     ax.matshow(attention[h])
        #     ax.set_xticks(range(len(top_tokens)))
        #     ax.set_xticklabels(top_tokens, rotation=90)
        #     if h == 0:
        #         ax.set_yticks(range(len(bottom_tokens)))
        #         ax.set_yticklabels(bottom_tokens)
        #     else:
        #         ax.set_yticklabels([])
        #     ax.set_xlabel(f'HEAD{h+1}')

        plt.margins(x=200)
        plt.show()

    def resize_token(self, tokens):
        new_tokens = []
        for i in range(0, len(tokens), 3):
            token = ' '.join(tokens[i:i+3])
            new_tokens.append(token)

        return new_tokens


if __name__ == '__main__':
    draw_attention = DrawAttention('aknos', 'O2')
    draw_attention.modelv2_plot()
    # draw_attention.get_first_top_attention()
