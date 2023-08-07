import tensorflow as tf
from tensorflow.keras import layers

'''
本模块定义asm2vec方法的基类
'''

# 负采样数目
num_ns = 2


class Asm2Vec(tf.keras.Model):
    """
    word2vec方式做词嵌入的模型
    """
    def __init__(self, vocab_size, embedding_dim):
        super(Asm2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="embed_layer")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns + 1)

    def call(self, inputs, training=None, mask=None):
        target, context = inputs
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


class Asm2VecLSTM(tf.keras.Model):
    """
    lstm方式做词嵌入的模型
    """
    def __init__(self, vocab_size, embedding_dim):
        super(Asm2VecLSTM, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size,
                                                embedding_dim,
                                                input_length=1,
                                                name="embed_layer")

        self.lstm_proj = tf.keras.Sequential(
            [layers.Bidirectional(layers.LSTM(32)),
             layers.BatchNormalization(),
             layers.Dense(1, activation='sigmoid'), ]
        )

    def call(self, inputs, training=None, mask=None):
        embed_output = self.embedding_layer(inputs)
        lstm_output = self.lstm_proj(embed_output)
        return lstm_output
