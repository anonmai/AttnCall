import tensorflow as tf
from transformer.encoder import EncoderLayer
from transformer.positionalembedding import PositionalEmbedding


class ModelV1Part1(EncoderLayer):
    """
    ModelV1的主体部分现在直接用的encoder layer
    """
    pass


class ModelV1(tf.keras.Model):
    """
    model_v1,这个model简单的由3个部分组成
    1.embedding层
    2.encoder层
    3.全联接层，输出0或1
    """
    def __init__(self,
                 *,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 vocab_size,
                 dropout_rate=0):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 第1部分，embedding
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, output_dim=d_model, pos_dim=41)

        # 第2部分，encoder
        self.enc_layers = [
            ModelV1Part1(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        # dropout层暂时未用上，为以后的实验准备
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.final_layer = tf.keras.layers.Dense(1)
        # 第3部分，全联接层输出结果
        # 3.1 这时候输出的还是一个shape为 (batch, MAX_LENGTH, embed_dim) 的张量，通过这一步变为 (batch, MAX_LENGTH, 1)
        # self.flat_layer = tf.keras.layers.Dense(1)
        # 3.2 再把它变为 (batch, 1)的结果，去和label比较
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, *args, **kwargs):
        # 这一步的输入为 (batch, MAX_LENGTH)
        print('x0 shape = ', x.shape)        # x0 shape = (None, 200)
        # 1. 经过这一步，会变为 (batch, NUM_INS * 3, embed的维度)
        x1 = self.pos_embedding(x)
        print('x1 shape = ', x1.shape)       # x1 shape = (None, 60, 5)

        # 2. dropout暂时没加，加上接口等以后可能会用到
        x2 = self.dropout(x1)
        print('x2 shape = ', x2.shape)       # x2 shape = (None, 60, 5)

        # 3. 所有的encoder layer层依次调用，x形状不变
        for i in range(self.num_layers):
            x2 = self.enc_layers[i](x2)
        x3 = x2
        print('x3 shape = ', x3.shape)       # x3 shape = (None, 60, 5)

        # 4. 把输出变为0/1
        x4 = tf.reshape(x3, [-1, x3.shape[1] * x3.shape[2]])
        print('x4 shape = ', x4.shape)       # x4 shape = (None, 300)
        # x = self.flat_layer(x)
        # x = tf.squeeze(x, axis=-1)
        result = self.final_layer(x4)
        print('result shape = ', result.shape)  # result shape = (None, 1)
        return result
