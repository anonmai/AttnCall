import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    embedding层
    """
    def __init__(self, vocab_size, pos_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.insn_length = 12
        self.insn_count = 10
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, output_dim, mask_zero=True)
        self.pos_embedding = tf.keras.layers.Embedding(pos_dim, output_dim, mask_zero=True)

    # def compute_mask(self, inputs, mask=None):
    #     """
    #     因为inputs既包含了word，又包含了position，所以mask需要把inputs拆分后再进行计算
    #     """
    #     # word = tf.squeeze(inputs[:, :1, :], axis=1)
    #     word = inputs
    #     mask = self.word_embedding.compute_mask(word)
    #
    #     return mask

    def call(self, inputs, *args, **kwargs):
        # 输入的inputs的shape应该为(batch_size, None)
        word = inputs
        embedding_output = self.word_embedding(word)
        x = tf.reshape(embedding_output, [-1, embedding_output.shape[1] // self.insn_length, self.insn_length, self.output_dim])
        pos_output = self.compute_pos(inputs)
        x = self.compute_index(x, pos_output)
        return x

    @tf.function
    def compute_index(self, x, pos_output):
        # x的shape为(batch, sequence_length//10, 12, output_dim)
        # 用2,4,4的方式拆分
        x1 = tf.reduce_sum(x[:, :, :2, :], axis=2, keepdims=True)
        x2 = tf.reduce_sum(x[:, :, 2:7, :], axis=2, keepdims=True)
        x3 = tf.reduce_sum(x[:, :, 7:, :], axis=2, keepdims=True)

        # 原本的pos向量shape是(sequence_length//12, output_dim)
        # 为了可以与x1, x2, x3相加，需要增加dim变为(1, sequence_length//12, 1, output_dim)
        pos = pos_output[tf.newaxis, :, tf.newaxis, :]

        # 这里用(batch, sequence_length//10, 1, output_dim)和(1, sequence_length//12, 12, output_dim)相加，后面的tensor会广播
        x1 = tf.add(x1, pos)
        x2 = tf.add(x2, pos)
        x3 = tf.add(x3, pos)

        # 再合并，现在的shape是(batch, None, 3, 5)
        x = tf.concat([x1, x2, x3], axis=2)
        x = tf.reshape(x, [-1, x.shape[1] * 3, self.output_dim])
        return x

    @tf.function
    def compute_pos(self, x):
        # x是input，shape为(batch, sequence_length)
        # 需要用到sequence_length,构造1-sequence_length//12的tensor
        # 起点从1开始，所以需要+1
        pos = tf.range(1, x.shape[1] // self.insn_length + 1)
        pos_output = self.pos_embedding(pos)
        return pos_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'insn_count': self.insn_count,
            'insn_length': self.insn_length,
            'output_dim': self.output_dim
        })
        return config
